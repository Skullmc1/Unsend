package com.unsend.keyboard.ime

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView
import kotlin.math.roundToInt

/**
 * Lightweight, 60/120fps-friendly QWERTY & Symbols keyboard view with haptic feedback.
 */
class KeyboardView(context: Context) : LinearLayout(context) {

    interface Listener {
        fun onCharacter(text: String)
        fun onBackspace()
        fun onEnter()
    }

    var listener: Listener? = null

    private enum class Layer { QWERTY, SYMBOLS, SYMBOLS_EXT }
    private enum class ShiftState { OFF, SINGLE, CAPS_LOCK }

    private var currentLayer = Layer.QWERTY
    private var shiftState = ShiftState.OFF
    private var lastShiftTapMs = 0L
    private var enterActionLabel = "⏎"

    private val repeatHandler = Handler(Looper.getMainLooper())
    private val backspaceRepeater = object : Runnable {
        override fun run() {
            listener?.onBackspace()
            repeatHandler.postDelayed(this, BACKSPACE_REPEAT_INTERVAL_MS)
        }
    }

    init {
        orientation = VERTICAL
        setBackgroundColor(COLOR_KEYBOARD_BG)
        setPadding(dp(3), dp(6), dp(3), dp(8))
        isHapticFeedbackEnabled = true
        rebuildKeys()
    }

    fun updateImeOptions(imeOptions: Int) {
        val action = imeOptions and EditorInfo.IME_MASK_ACTION
        val newLabel = when (action) {
            EditorInfo.IME_ACTION_SEND -> "Send"
            EditorInfo.IME_ACTION_SEARCH -> "Search"
            EditorInfo.IME_ACTION_GO -> "Go"
            EditorInfo.IME_ACTION_DONE -> "Done"
            EditorInfo.IME_ACTION_NEXT -> "Next"
            else -> "⏎"
        }
        if (newLabel != enterActionLabel) {
            enterActionLabel = newLabel
            rebuildKeys()
        }
    }

    private fun rebuildKeys() {
        removeAllViews()
        when (currentLayer) {
            Layer.QWERTY -> buildQwertyRows()
            Layer.SYMBOLS -> buildSymbolsRows()
            Layer.SYMBOLS_EXT -> buildExtendedSymbolsRows()
        }
        addBottomSpacerBar()
    }

    private fun addBottomSpacerBar() {
        val barContainer = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(COLOR_KEYBOARD_BG)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(26))
        }
        val subtlePill = android.view.View(context).apply {
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E1E1E"))
                cornerRadius = dp(2).toFloat()
            }
            layoutParams = LayoutParams(dp(38), dp(3))
        }
        barContainer.addView(subtlePill)
        addView(barContainer)
    }

    private fun buildQwertyRows() {
        val row1 = listOf("q", "w", "e", "r", "t", "y", "u", "i", "o", "p")
        val row2 = listOf("a", "s", "d", "f", "g", "h", "j", "k", "l")
        val row3 = listOf("z", "x", "c", "v", "b", "n", "m")

        addRow(row1.map { ch -> KeySpec.CharKey(applyShift(ch)) })

        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(16), 0, dp(16), 0)
            row2.forEach { ch ->
                addView(createKeyView(KeySpec.CharKey(applyShift(ch)), weight = 1f))
            }
        }, rowLayoutParams())

        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            val shiftLabel = when (shiftState) {
                ShiftState.OFF -> "⇧"
                ShiftState.SINGLE -> "⬆"
                ShiftState.CAPS_LOCK -> "⇪"
            }
            addView(
                createKeyView(
                    KeySpec.ActionKey(
                        label = shiftLabel,
                        isActive = shiftState != ShiftState.OFF,
                        action = ::handleShiftTap,
                    ),
                    weight = 1.45f,
                ),
            )
            row3.forEach { ch ->
                addView(createKeyView(KeySpec.CharKey(applyShift(ch)), weight = 1f))
            }
            addView(createBackspaceKey(weight = 1.45f))
        }, rowLayoutParams())

        addBottomRow(switchLabel = "?123", switchTarget = Layer.SYMBOLS, leftChar = ",", rightChar = ".")
    }

    private fun buildSymbolsRows() {
        addRow(listOf("1", "2", "3", "4", "5", "6", "7", "8", "9", "0").map { KeySpec.CharKey(it) })
        addRow(listOf("@", "#", "$", "_", "&", "-", "+", "(", ")", "/").map { KeySpec.CharKey(it) })

        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            addView(
                createKeyView(
                    KeySpec.ActionKey(label = "=\\<", action = {
                        currentLayer = Layer.SYMBOLS_EXT
                        rebuildKeys()
                    }),
                    weight = 1.45f,
                ),
            )
            listOf("*", "\"", "'", ":", ";", "!", "?").forEach { sym ->
                addView(createKeyView(KeySpec.CharKey(sym), weight = 1f))
            }
            addView(createBackspaceKey(weight = 1.45f))
        }, rowLayoutParams())

        addBottomRow(switchLabel = "ABC", switchTarget = Layer.QWERTY, leftChar = ",", rightChar = ".")
    }

    private fun buildExtendedSymbolsRows() {
        addRow(listOf("~", "`", "|", "•", "√", "π", "÷", "×", "§", "Δ").map { KeySpec.CharKey(it) })
        addRow(listOf("£", "¢", "€", "¥", "^", "°", "=", "{", "}", "\\").map { KeySpec.CharKey(it) })

        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            addView(
                createKeyView(
                    KeySpec.ActionKey(label = "?123", action = {
                        currentLayer = Layer.SYMBOLS
                        rebuildKeys()
                    }),
                    weight = 1.45f,
                ),
            )
            listOf("%", "©", "®", "™", "✓", "[", "]").forEach { sym ->
                addView(createKeyView(KeySpec.CharKey(sym), weight = 1f))
            }
            addView(createBackspaceKey(weight = 1.45f))
        }, rowLayoutParams())

        addBottomRow(switchLabel = "ABC", switchTarget = Layer.QWERTY, leftChar = "<", rightChar = ">")
    }

    private fun addBottomRow(switchLabel: String, switchTarget: Layer, leftChar: String, rightChar: String) {
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            addView(
                createKeyView(
                    KeySpec.ActionKey(label = switchLabel, action = {
                        currentLayer = switchTarget
                        rebuildKeys()
                    }),
                    weight = 1.5f,
                ),
            )
            addView(createKeyView(KeySpec.CharKey(leftChar), weight = 1f, isSpecialBg = true))
            addView(
                createKeyView(
                    KeySpec.ActionKey(label = "UNSEND", isSpace = true, action = {
                        listener?.onCharacter(" ")
                    }),
                    weight = 4.2f,
                ),
            )
            addView(createKeyView(KeySpec.CharKey(rightChar), weight = 1f, isSpecialBg = true))
            addView(
                createKeyView(
                    KeySpec.ActionKey(label = enterActionLabel, isPrimary = true, action = {
                        listener?.onEnter()
                    }),
                    weight = 1.6f,
                ),
            )
        }, rowLayoutParams())
    }

    private fun addRow(keys: List<KeySpec>) {
        val row = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER
            keys.forEach { spec ->
                addView(createKeyView(spec, weight = 1f))
            }
        }
        addView(row, rowLayoutParams())
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun createKeyView(
        spec: KeySpec,
        weight: Float,
        isSpecialBg: Boolean = false,
    ): TextView {
        val baseColor = when {
            spec is KeySpec.ActionKey && spec.isPrimary -> COLOR_KEY_PRIMARY
            spec is KeySpec.ActionKey && spec.isActive -> COLOR_KEY_ACTIVE
            spec is KeySpec.ActionKey && !spec.isSpace -> COLOR_KEY_SPECIAL
            isSpecialBg -> COLOR_KEY_SPECIAL
            else -> COLOR_KEY_NORMAL
        }
        val textColor = when {
            spec is KeySpec.ActionKey && spec.isPrimary -> COLOR_KEYBOARD_BG
            spec is KeySpec.ActionKey && spec.isActive -> COLOR_KEYBOARD_BG
            spec is KeySpec.ActionKey && spec.isSpace -> COLOR_TEXT_MUTED
            spec is KeySpec.ActionKey -> COLOR_TEXT_MUTED
            isSpecialBg -> COLOR_TEXT_MUTED
            else -> COLOR_TEXT_PRIMARY
        }

        return TextView(context).apply {
            text = spec.label
            gravity = Gravity.CENTER
            setTextColor(textColor)
            setTextSize(
                TypedValue.COMPLEX_UNIT_SP,
                when {
                    spec is KeySpec.ActionKey && spec.isSpace -> 10.5f
                    spec.label.length > 2 -> 12f
                    else -> 19f
                },
            )
            if (spec is KeySpec.ActionKey && spec.isSpace) {
                letterSpacing = 0.18f
            }
            typeface = if (spec is KeySpec.ActionKey && spec.isPrimary) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            background = keyDrawable(baseColor)
            layoutParams = LayoutParams(0, dp(46), weight).apply {
                setMargins(dp(2), dp(3), dp(2), dp(3))
            }

            setOnTouchListener { view, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        view.background = keyDrawable(COLOR_KEY_PRESSED)
                        view.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                        true
                    }
                    MotionEvent.ACTION_UP -> {
                        view.background = keyDrawable(baseColor)
                        when (spec) {
                            is KeySpec.CharKey -> {
                                listener?.onCharacter(spec.label)
                                if (shiftState == ShiftState.SINGLE) {
                                    shiftState = ShiftState.OFF
                                    rebuildKeys()
                                }
                            }
                            is KeySpec.ActionKey -> spec.action()
                        }
                        true
                    }
                    MotionEvent.ACTION_CANCEL -> {
                        view.background = keyDrawable(baseColor)
                        true
                    }
                    else -> false
                }
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun createBackspaceKey(weight: Float): TextView {
        return TextView(context).apply {
            text = "⌫"
            gravity = Gravity.CENTER
            setTextColor(COLOR_TEXT_PRIMARY)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            background = keyDrawable(COLOR_KEY_SPECIAL)
            layoutParams = LayoutParams(0, dp(46), weight).apply {
                setMargins(dp(2), dp(3), dp(2), dp(3))
            }

            setOnTouchListener { view, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        view.background = keyDrawable(COLOR_KEY_PRESSED)
                        view.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                        listener?.onBackspace()
                        repeatHandler.postDelayed(backspaceRepeater, BACKSPACE_INITIAL_DELAY_MS)
                        true
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        view.background = keyDrawable(COLOR_KEY_SPECIAL)
                        repeatHandler.removeCallbacks(backspaceRepeater)
                        true
                    }
                    else -> false
                }
            }
        }
    }

    private fun handleShiftTap() {
        val now = System.currentTimeMillis()
        shiftState = when (shiftState) {
            ShiftState.OFF -> ShiftState.SINGLE
            ShiftState.SINGLE -> if (now - lastShiftTapMs < 350L) ShiftState.CAPS_LOCK else ShiftState.OFF
            ShiftState.CAPS_LOCK -> ShiftState.OFF
        }
        lastShiftTapMs = now
        rebuildKeys()
    }

    private fun applyShift(ch: String): String =
        if (shiftState != ShiftState.OFF) ch.uppercase() else ch

    private fun rowLayoutParams(): LayoutParams =
        LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)

    private fun keyDrawable(color: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(3).toFloat()
        }

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            resources.displayMetrics,
        ).roundToInt()

    private sealed class KeySpec(val label: String) {
        class CharKey(label: String) : KeySpec(label)
        class ActionKey(
            label: String,
            val isPrimary: Boolean = false,
            val isActive: Boolean = false,
            val isSpace: Boolean = false,
            val action: () -> Unit,
        ) : KeySpec(label)
    }

    companion object {
        private const val BACKSPACE_INITIAL_DELAY_MS = 380L
        private const val BACKSPACE_REPEAT_INTERVAL_MS = 55L

        private val COLOR_KEYBOARD_BG = Color.parseColor("#000000")
        private val COLOR_KEY_NORMAL = Color.parseColor("#0D0D0D")
        private val COLOR_KEY_SPECIAL = Color.parseColor("#050505")
        private val COLOR_KEY_ACTIVE = Color.parseColor("#FFFFFF")
        private val COLOR_KEY_PRIMARY = Color.parseColor("#FFFFFF")
        private val COLOR_KEY_PRESSED = Color.parseColor("#262626")
        private val COLOR_TEXT_PRIMARY = Color.parseColor("#F5F5F5")
        private val COLOR_TEXT_MUTED = Color.parseColor("#666666")
    }
}
