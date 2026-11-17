package com.unsend.keyboard.ml

import kotlin.math.exp

/**
 * Communication context presets that adjust how Laya frames escalation/regret risk.
 */
enum class ContextPreset(
    val label: String,
    val amberThreshold: Float,
    val redThreshold: Float,
) {
    WORK(label = "Work", amberThreshold = 0.35f, redThreshold = 0.65f),
    PERSONAL(label = "Personal", amberThreshold = 0.45f, redThreshold = 0.75f),
    DATING(label = "Dating", amberThreshold = 0.40f, redThreshold = 0.70f),
    NEGOTIATION(label = "Negotiation", amberThreshold = 0.38f, redThreshold = 0.68f);

    fun next(): ContextPreset {
        val all = entries
        return all[(ordinal + 1) % all.size]
    }
}

enum class ToneLabel(val displayName: String) {
    WARM("Warm"),
    ASSERTIVE("Assertive"),
    DEFENSIVE("Defensive"),
    PASSIVE_AGGRESSIVE("Passive-Aggressive"),
    BLUNT("Blunt"),
    UNCLEAR("Unclear"),
}

enum class RiskLevel {
    GREEN,
    AMBER,
    RED,
}

enum class ModelStatus {
    READY_ONNX,
    PREVIEW_MODE,
    NOT_CONFIGURED,
}

/**
 * Result of a single-forward-pass Laya evaluation.
 */
data class LayaAnalysis(
    val escalationRisk: Float,
    val riskLevel: RiskLevel,
    val primaryTone: ToneLabel,
    val toneProbabilities: Map<ToneLabel, Float>,
    val clarityScore: Float,
    val answeredQuestionScore: Float?,
    val inferenceTimeMs: Long,
    val isPreviewHeuristic: Boolean,
)

object ToneSchema {
    const val MIN_DRAFT_LENGTH = 15

    fun classifyRisk(risk: Float, preset: ContextPreset): RiskLevel = when {
        risk >= preset.redThreshold -> RiskLevel.RED
        risk >= preset.amberThreshold -> RiskLevel.AMBER
        else -> RiskLevel.GREEN
    }

    fun buildPrompt(
        draft: String,
        incomingReply: String?,
        preset: ContextPreset,
    ): String = buildString {
        append("[CONTEXT: ").append(preset.label).append("] ")
        if (!incomingReply.isNullOrBlank()) {
            append("[INCOMING: ").append(incomingReply.trim()).append("] ")
        }
        append("[DRAFT: ").append(draft.trim()).append("]")
    }

    fun sigmoid(x: Float): Float = (1.0f / (1.0f + exp(-x.coerceIn(-20f, 20f))))

    fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return FloatArray(0)
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { i -> exp((logits[i] - maxLogit).coerceIn(-20f, 20f)) }
        val sum = exps.sum().coerceAtLeast(1e-6f)
        return FloatArray(logits.size) { i -> exps[i] / sum }
    }
}
