package com.unsend.keyboard.ime

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.unsend.keyboard.ml.ContextPreset
import com.unsend.keyboard.ml.LayaAnalysis
import com.unsend.keyboard.ml.ModelStatus
import com.unsend.keyboard.ml.RiskLevel
import com.unsend.keyboard.ml.ToneLabel
import kotlin.math.roundToInt

/**
 * Swiss Editorial 48dp HUD toolbar + expandable hairline telemetry drawer in pure #000000 OLED black.
 */
class HudBarView(context: Context) : LinearLayout(context) {

    var onCyclePreset: (() -> Unit)? = null
    var onStatusClick: (() -> Unit)? = null
    var onToggleReplyCheck: (() -> Unit)? = null
    var onClearReplyContext: (() -> Unit)? = null
    var onOpenSetup: (() -> Unit)? = null

    private val presetLabel: TextView
    private val statusContainer: LinearLayout
    private val statusDot: View
    private val statusLabel: TextView
    private val replyChip: TextView
    private val expandButton: TextView

    private val drawerContainer: LinearLayout
    private val replyPreviewRow: LinearLayout
    private val replyPreviewText: TextView
    private val signalTitle: TextView
    private val signalPct: TextView
    private val riskBar: ProgressBar
    private val metricsLine: TextView
    private val toneRows: Map<ToneLabel, ToneTextRow>

    private var isDrawerExpanded = false
    private var currentStatus: ModelStatus = ModelStatus.NOT_CONFIGURED

    init {
        orientation = VERTICAL
        setBackgroundColor(COLOR_BLACK)

        // 1. Expandable Swiss Editorial Drawer
        drawerContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            visibility = GONE
            setBackgroundColor(COLOR_BLACK)
            setPadding(dp(18), dp(12), dp(18), dp(12))
        }

        // Top hairline divider
        drawerContainer.addView(hairlineView())

        // Incoming reply quote
        replyPreviewRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = GONE
            setBackgroundColor(COLOR_SUBTLE_SURFACE)
            setPadding(dp(10), dp(8), dp(10), dp(8))
        }
        replyPreviewText = TextView(context).apply {
            setTextColor(COLOR_TEXT_SECONDARY)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        }
        val clearReplyBtn = TextView(context).apply {
            text = "CLEAR"
            setTextColor(COLOR_WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
            letterSpacing = 0.12f
            setPadding(dp(10), dp(2), dp(2), dp(2))
            setOnClickListener { onClearReplyContext?.invoke() }
        }
        replyPreviewRow.addView(replyPreviewText)
        replyPreviewRow.addView(clearReplyBtn)
        drawerContainer.addView(
            replyPreviewRow,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(8)
                bottomMargin = dp(8)
            },
        )

        // Editorial primary signal header
        val microHeader = TextView(context).apply {
            text = "PRIMARY SIGNAL"
            setTextColor(COLOR_TEXT_MUTED)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
            letterSpacing = 0.14f
            setPadding(0, dp(8), 0, dp(2))
        }
        drawerContainer.addView(microHeader)

        val signalRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.BOTTOM
        }
        signalTitle = TextView(context).apply {
            text = "Mirror Ready"
            typeface = Typeface.SERIF
            setTextColor(COLOR_WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        }
        signalPct = TextView(context).apply {
            text = "0%"
            typeface = Typeface.SERIF
            setTextColor(COLOR_WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            gravity = Gravity.END
        }
        signalRow.addView(signalTitle)
        signalRow.addView(signalPct)
        drawerContainer.addView(signalRow)

        // 2dp Hairline Risk Bar
        riskBar = ProgressBar(
            context,
            null,
            android.R.attr.progressBarStyleHorizontal,
        ).apply {
            max = 100
            progress = 0
            progressBackgroundTintList = android.content.res.ColorStateList.valueOf(COLOR_HAIRLINE)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(3)).apply {
                topMargin = dp(6)
                bottomMargin = dp(8)
            }
        }
        drawerContainer.addView(riskBar)

        // Summary metrics line: Clarity • Answered • Latency
        metricsLine = TextView(context).apply {
            text = "Clarity —   •   100% offline"
            setTextColor(COLOR_TEXT_MUTED)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setPadding(0, 0, 0, dp(8))
        }
        drawerContainer.addView(metricsLine)

        // 2-column Tone Breakdown
        val toneGrid = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(0, dp(4), 0, 0)
        }
        val toneMap = mutableMapOf<ToneLabel, ToneTextRow>()
        val labels = ToneLabel.entries
        for (i in labels.indices step 2) {
            val pairRow = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            val leftLabel = labels[i]
            val leftRow = ToneTextRow(context, leftLabel.displayName)
            toneMap[leftLabel] = leftRow
            pairRow.addView(
                leftRow.root,
                LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(16) },
            )
            if (i + 1 < labels.size) {
                val rightLabel = labels[i + 1]
                val rightRow = ToneTextRow(context, rightLabel.displayName)
                toneMap[rightLabel] = rightRow
                pairRow.addView(rightRow.root, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            }
            toneGrid.addView(pairRow)
        }
        toneRows = toneMap
        drawerContainer.addView(toneGrid)
        addView(drawerContainer, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))

        // Top hairline above toolbar
        addView(hairlineView())

        // 2. Compact 48dp Swiss Editorial Toolbar
        val toolbar = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(6), dp(14), dp(6))
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(48))
        }

        presetLabel = TextView(context).apply {
            text = "WORK"
            setTextColor(COLOR_TEXT_SECONDARY)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            letterSpacing = 0.12f
            setPadding(dp(4), dp(6), dp(10), dp(6))
            setOnClickListener { onCyclePreset?.invoke() }
        }

        statusContainer = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(6), dp(6), dp(6), dp(6))
            setOnClickListener {
                if (currentStatus == ModelStatus.NOT_CONFIGURED) {
                    onOpenSetup?.invoke()
                } else {
                    onStatusClick?.invoke()
                }
            }
        }
        statusDot = View(context).apply {
            background = circleDrawable(COLOR_NEUTRAL)
            layoutParams = LayoutParams(dp(6), dp(6)).apply {
                marginEnd = dp(7)
            }
        }
        statusLabel = TextView(context).apply {
            text = "MIRROR READY"
            setTextColor(COLOR_WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            letterSpacing = 0.10f
            setTypeface(null, Typeface.BOLD)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
        }
        statusContainer.addView(statusDot)
        statusContainer.addView(statusLabel)

        replyChip = TextView(context).apply {
            text = "REPLY"
            setTextColor(COLOR_TEXT_MUTED)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            letterSpacing = 0.12f
            setPadding(dp(10), dp(6), dp(8), dp(6))
            setOnClickListener { onToggleReplyCheck?.invoke() }
        }

        expandButton = TextView(context).apply {
            text = "▾"
            setTextColor(COLOR_TEXT_MUTED)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(4), dp(2), dp(4))
            setOnClickListener { toggleDrawer() }
        }

        toolbar.addView(presetLabel)
        toolbar.addView(statusContainer, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
        toolbar.addView(replyChip)
        toolbar.addView(expandButton)
        addView(toolbar)

        // Bottom hairline below toolbar
        addView(hairlineView())
    }

    fun toggleDrawer() {
        isDrawerExpanded = !isDrawerExpanded
        drawerContainer.visibility = if (isDrawerExpanded) VISIBLE else GONE
        expandButton.text = if (isDrawerExpanded) "▴" else "▾"
    }

    fun updatePreset(preset: ContextPreset) {
        presetLabel.text = preset.label.uppercase()
    }

    fun updateReplyContext(incomingReply: String?) {
        val active = !incomingReply.isNullOrBlank()
        replyChip.setTextColor(if (active) COLOR_WHITE else COLOR_TEXT_MUTED)
        replyChip.paintFlags = if (active) {
            replyChip.paintFlags or Paint.UNDERLINE_TEXT_FLAG
        } else {
            replyChip.paintFlags and Paint.UNDERLINE_TEXT_FLAG.inv()
        }

        if (active) {
            replyPreviewRow.visibility = VISIBLE
            replyPreviewText.text = "Replying to: \"${incomingReply.trim()}\""
        } else {
            replyPreviewRow.visibility = GONE
        }
    }

    fun renderState(
        modelStatus: ModelStatus,
        analysis: LayaAnalysis?,
        isEvaluating: Boolean,
        draftLength: Int,
    ) {
        currentStatus = modelStatus

        if (modelStatus == ModelStatus.NOT_CONFIGURED) {
            statusDot.background = circleDrawable(COLOR_AMBER)
            statusLabel.text = "FINISH SETUP"
            signalTitle.text = "Setup Required"
            signalPct.text = "—"
            metricsLine.text = "Import Laya .onnx or enable Preview Mode in Unsend."
            return
        }

        if (isEvaluating) {
            statusDot.background = circleDrawable(COLOR_WHITE)
            statusLabel.text = "EVALUATING…"
            return
        }

        if (analysis == null) {
            statusDot.background = circleDrawable(COLOR_NEUTRAL)
            statusLabel.text = if (draftLength in 1 until 15) "KEEP TYPING…" else "MIRROR READY"
            return
        }

        val accentColor = when (analysis.riskLevel) {
            RiskLevel.GREEN -> COLOR_GREEN
            RiskLevel.AMBER -> COLOR_AMBER
            RiskLevel.RED -> COLOR_RED
        }
        statusDot.background = circleDrawable(accentColor)

        val riskPct = (analysis.escalationRisk * 100f).roundToInt()
        val toneUpper = analysis.primaryTone.displayName.uppercase()
        statusLabel.text = "$riskPct% $toneUpper"

        signalTitle.text = when (analysis.riskLevel) {
            RiskLevel.GREEN -> "${analysis.primaryTone.displayName}, Clear"
            RiskLevel.AMBER -> "${analysis.primaryTone.displayName} Tone"
            RiskLevel.RED -> "${analysis.primaryTone.displayName}, High Risk"
        }
        signalPct.text = "$riskPct%"
        signalPct.setTextColor(accentColor)

        riskBar.progress = riskPct
        riskBar.progressTintList = android.content.res.ColorStateList.valueOf(accentColor)

        val clarityPct = (analysis.clarityScore * 100f).roundToInt()
        val answeredPart = analysis.answeredQuestionScore?.let {
            "   •   Answered ${(it * 100f).roundToInt()}%"
        }.orEmpty()
        val engineTag = if (analysis.isPreviewHeuristic) "Preview" else "Laya"
        metricsLine.text = "Clarity $clarityPct%$answeredPart   •   $engineTag ${analysis.inferenceTimeMs}ms"

        toneRows.forEach { (tone, row) ->
            val prob = analysis.toneProbabilities[tone] ?: 0f
            row.update(prob, isPrimary = tone == analysis.primaryTone)
        }
    }

    private fun hairlineView(): View = View(context).apply {
        setBackgroundColor(COLOR_HAIRLINE)
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(1).coerceAtLeast(1))
    }

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            resources.displayMetrics,
        ).roundToInt()

    private fun circleDrawable(color: Int): GradientDrawable =
        GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(color)
        }

    private inner class ToneTextRow(context: Context, label: String) {
        val root: LinearLayout = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(2), 0, dp(2))
        }
        private val titleView = TextView(context).apply {
            text = label
            setTextColor(COLOR_TEXT_MUTED)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        }
        private val pctView = TextView(context).apply {
            text = "0%"
            setTextColor(COLOR_TEXT_SECONDARY)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            typeface = Typeface.MONOSPACE
            gravity = Gravity.END
        }

        init {
            root.addView(titleView)
            root.addView(pctView)
        }

        fun update(probability: Float, isPrimary: Boolean) {
            val pct = (probability.coerceIn(0f, 1f) * 100f).roundToInt()
            pctView.text = "$pct%"
            titleView.setTextColor(if (isPrimary) COLOR_WHITE else COLOR_TEXT_MUTED)
            pctView.setTextColor(if (isPrimary) COLOR_WHITE else COLOR_TEXT_SECONDARY)
            pctView.setTypeface(Typeface.MONOSPACE, if (isPrimary) Typeface.BOLD else Typeface.NORMAL)
        }
    }

    companion object {
        private val COLOR_BLACK = Color.parseColor("#000000")
        private val COLOR_SUBTLE_SURFACE = Color.parseColor("#0A0A0A")
        private val COLOR_HAIRLINE = Color.parseColor("#1C1C1C")
        private val COLOR_WHITE = Color.parseColor("#FAFAFA")
        private val COLOR_TEXT_SECONDARY = Color.parseColor("#A3A3A3")
        private val COLOR_TEXT_MUTED = Color.parseColor("#666666")
        private val COLOR_GREEN = Color.parseColor("#22C55E")
        private val COLOR_AMBER = Color.parseColor("#F59E0B")
        private val COLOR_RED = Color.parseColor("#EF4444")
        private val COLOR_NEUTRAL = Color.parseColor("#525252")
    }
}
