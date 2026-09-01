import { afterEach, describe, expect, test } from "bun:test"

import { releaseAllPromptAsyncReservationsForTesting } from "../../shared/prompt-async-gate"
import { setPromptReservation } from "@oh-my-opencode/utils/prompt-async-gate/reservations"
import { createAutoRetryHelpers } from "./auto-retry"
import { createFallbackState } from "./fallback-state"
import { SessionCategoryRegistry } from "../../shared/session-category-registry"
import { installRuntimeFallbackTestClock, restoreRuntimeFallbackTestClock } from "./test-timeout-clock.test-support"
import type { HookDeps, RuntimeFallbackPluginInput } from "./types"

const SESSION_TIMEOUT_MS = 30

type GateHarness = {
  readonly ctx: RuntimeFallbackPluginInput
  readonly promptModels: string[]
  readonly abortSources: string[]
  readonly toastTitles: string[]
  sessionStatus: "idle" | "busy"
}

function createGateHarness(): GateHarness {
  const promptModels: string[] = []
  const abortSources: string[] = []
  const toastTitles: string[] = []
  const harness = {
    promptModels,
    abortSources,
    toastTitles,
    sessionStatus: "idle" as "idle" | "busy",
  }

  const ctx: RuntimeFallbackPluginInput = {
    client: {
      session: {
        abort: async () => {
          abortSources.push("abort")
          return {}
        },
        messages: async () => ({
          data: [
            {
              info: { role: "user" },
              parts: [{ type: "text", text: "retry this" }],
            },
          ],
        }),
        promptAsync: async (input) => {
          promptModels.push(`${input.body.model.providerID}/${input.body.model.modelID}`)
          return {}
        },
        status: async () => ({
          data: { [SESSION_ID]: { type: harness.sessionStatus === "busy" ? "busy" : "idle" } },
        }),
      },
      tui: {
        showToast: async (input) => {
          toastTitles.push(input.body.title)
          return {}
        },
      },
    },
    directory: "/test/dir",
  }

  return { ...harness, ctx } as GateHarness
}

const SESSION_ID = "session-timeout-real-gate"

function createDeps(harness: GateHarness): HookDeps {
  return {
    ctx: harness.ctx,
    config: {
      enabled: true,
      retry_on_errors: [429, 503, 529],
      max_fallback_attempts: 2,
      cooldown_seconds: 60,
      timeout_seconds: 30,
      notify_on_fallback: true,
      restore_primary_after_cooldown: false,
    },
    options: {
      session_timeout_ms: SESSION_TIMEOUT_MS,
    },
    pluginConfig: {
      categories: {
        test: {
          fallback_models: ["litellm/openai.eu.gpt-5.5", "google/gemini-2.5-pro"],
        },
      },
    },
    sessionStates: new Map(),
    sessionLastAccess: new Map(),
    sessionRetryInFlight: new Set(),
    sessionAwaitingFallbackResult: new Set(),
    sessionFallbackTimeouts: new Map(),
    sessionStatusRetryKeys: new Map(),
    internallyAbortedSessions: new Set(),
  }
}

// Holds the session reservation past every reserved-retry backoff the dispatcher
// performs (6 attempts, 500ms..3000ms linear), so the gate keeps answering
// "reserved" for the whole dispatch instead of racing wall-clock time.
const RESERVATION_HOLD_MS = 60_000

function reserveSession(sessionID: string, holdMs = RESERVATION_HOLD_MS): void {
  setPromptReservation(sessionID, {
    source: "user-prompt",
    dedupeKey: `reserved-${sessionID}`,
    reservedAt: Date.now(),
    token: Symbol(`reserved-${sessionID}`),
    expiresAt: Date.now() + holdMs,
  })
}

async function flushPromptGateMicrotasks(): Promise<void> {
  for (let index = 0; index < 10; index += 1) {
    await Promise.resolve()
  }
}

// The timeout callback awaits the dispatcher, whose reserved-retry backoff
// awaits further timers. A single awaited advance would deadlock on itself, so
// advances are issued without awaiting and the loop stops on an observed
// condition rather than on elapsed wall-clock time.
async function pumpUntil(
  clock: { readonly advanceBy: (ms: number) => Promise<void> },
  condition: () => boolean,
  maxSteps = 400,
  stepMs = 250,
): Promise<boolean> {
  for (let step = 0; step < maxSteps; step += 1) {
    await flushPromptGateMicrotasks()
    if (condition()) return true
    void clock.advanceBy(stepMs)
  }
  await flushPromptGateMicrotasks()
  return condition()
}

describe("session timeout fallback through the real internal-prompt gate", () => {
  afterEach(() => {
    releaseAllPromptAsyncReservationsForTesting()
    SessionCategoryRegistry.clear()
    restoreRuntimeFallbackTestClock()
  })

  test("#given the real gate rejects every timeout dispatch as reserved #when the attempt cap is reached #then no prompt is sent, attempts are preserved, and the caller is failed terminally", async () => {
    // given
    SessionCategoryRegistry.register(SESSION_ID, "test")
    const harness = createGateHarness()
    const deps = createDeps(harness)
    const state = createFallbackState("openai/gpt-5.4")
    deps.sessionStates.set(SESSION_ID, state)
    deps.sessionAwaitingFallbackResult.add(SESSION_ID)
    const helpers = createAutoRetryHelpers(deps)
    reserveSession(SESSION_ID)
    const clock = installRuntimeFallbackTestClock()

    // when
    helpers.scheduleSessionFallbackTimeout(SESSION_ID, undefined)
    const reachedTerminalFailure = await pumpUntil(
      clock,
      () => harness.toastTitles.length > 0,
    )

    // then
    expect(reachedTerminalFailure).toBe(true)
    expect(harness.promptModels).toEqual([])
    expect(state.attemptCount).toBe(2)
    expect(state.currentModel).toBe("openai/gpt-5.4")
    expect(state.fallbackIndex).toBe(-1)
    expect(deps.sessionAwaitingFallbackResult.has(SESSION_ID)).toBe(false)
    expect(deps.sessionFallbackTimeouts.has(SESSION_ID)).toBe(false)
    expect(deps.sessionRetryInFlight.has(SESSION_ID)).toBe(false)
    expect(harness.toastTitles).toEqual(["Model Fallback Failed"])
  })

  test("#given the real gate reports the session active #when the timeout dispatch is queued and accepted #then exactly one prompt is sent for the accepted retry", async () => {
    // given
    SessionCategoryRegistry.register(SESSION_ID, "test")
    const harness = createGateHarness()
    harness.sessionStatus = "busy"
    const deps = createDeps(harness)
    // The queued dispatch drains on the gate's own retry interval. A timeout
    // shorter than that drain would re-arm and prepare a second attempt, which
    // is a separate retry rather than a duplicate of this one.
    deps.options = { session_timeout_ms: 5_000 }
    const state = createFallbackState("openai/gpt-5.4")
    deps.sessionStates.set(SESSION_ID, state)
    deps.sessionAwaitingFallbackResult.add(SESSION_ID)
    const helpers = createAutoRetryHelpers(deps)
    const clock = installRuntimeFallbackTestClock()

    // when
    helpers.scheduleSessionFallbackTimeout(SESSION_ID, undefined)
    const preparedFallback = await pumpUntil(clock, () => state.attemptCount > 0)
    harness.sessionStatus = "idle"
    const dispatched = await pumpUntil(clock, () => harness.promptModels.length > 0)
    helpers.clearSessionFallbackTimeout(SESSION_ID)
    await flushPromptGateMicrotasks()

    // then
    expect(preparedFallback).toBe(true)
    expect(dispatched).toBe(true)
    expect(harness.promptModels).toEqual(["litellm/openai.eu.gpt-5.5"])
    expect(state.attemptCount).toBe(1)
    expect(state.currentModel).toBe("litellm/openai.eu.gpt-5.5")
    expect(deps.sessionAwaitingFallbackResult.has(SESSION_ID)).toBe(true)
  })

  test("#given a timeout dispatch already ran for an older state generation #when a replacement generation exists #then the stale completion neither rewinds attempts nor dispatches again", async () => {
    // given
    SessionCategoryRegistry.register(SESSION_ID, "test")
    const harness = createGateHarness()
    const deps = createDeps(harness)
    const staleState = createFallbackState("openai/gpt-5.4")
    deps.sessionStates.set(SESSION_ID, staleState)
    deps.sessionAwaitingFallbackResult.add(SESSION_ID)
    const helpers = createAutoRetryHelpers(deps)
    reserveSession(SESSION_ID)
    const clock = installRuntimeFallbackTestClock()
    helpers.scheduleSessionFallbackTimeout(SESSION_ID, undefined)

    // when
    const replacementState = createFallbackState("google/gemini-2.5-pro")
    replacementState.attemptCount = 1
    deps.sessionStates.set(SESSION_ID, replacementState)
    await pumpUntil(clock, () => harness.promptModels.length > 0, 60)
    helpers.clearSessionFallbackTimeout(SESSION_ID)
    await flushPromptGateMicrotasks()

    // then
    expect(harness.promptModels).toEqual([])
    expect(replacementState.attemptCount).toBe(1)
    expect(replacementState.currentModel).toBe("google/gemini-2.5-pro")
    expect(replacementState.fallbackIndex).toBe(-1)
  })
})
