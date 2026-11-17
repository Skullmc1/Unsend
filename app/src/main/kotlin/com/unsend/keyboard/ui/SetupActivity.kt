package com.unsend.keyboard.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.ViewTreeObserver
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.unsend.keyboard.ime.UnsendImeService
import com.unsend.keyboard.ml.LayaEngine
import com.unsend.keyboard.ml.ModelStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class SetupActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color.White,
                    onPrimary = Color.Black,
                    background = Color.Black,
                    surface = Color.Black,
                    onSurface = Color(0xFFFAFAFA),
                    onSurfaceVariant = Color(0xFF737373),
                ),
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color.Black,
                ) {
                    SetupScreen()
                }
            }
        }
    }
}

@Composable
private fun SetupScreen() {
    val context = LocalContext.current
    val view = LocalView.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val layaEngine = remember { LayaEngine(context) }

    var isImeEnabled by remember { mutableStateOf(checkImeEnabled(context)) }
    var isImeSelected by remember { mutableStateOf(checkImeSelected(context)) }
    var modelStatus by remember { mutableStateOf(layaEngine.getStatus()) }
    var previewEnabled by remember { mutableStateOf(layaEngine.isPreviewModeEnabled()) }
    var onnxFileName by remember { mutableStateOf(layaEngine.findOnnxModelFile()?.name) }
    var hasVocab by remember { mutableStateOf(layaEngine.findVocabFile() != null) }
    var isImporting by remember { mutableStateOf(false) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableStateOf(0f) }
    var sandboxDraft by remember { mutableStateOf("") }

    fun refreshStatus() {
        isImeEnabled = checkImeEnabled(context)
        isImeSelected = checkImeSelected(context)
        modelStatus = layaEngine.getStatus()
        previewEnabled = layaEngine.isPreviewModeEnabled()
        onnxFileName = layaEngine.findOnnxModelFile()?.name
        hasVocab = layaEngine.findVocabFile() != null
    }

    DisposableEffect(lifecycleOwner, view) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                refreshStatus()
            }
        }
        val focusListener = ViewTreeObserver.OnWindowFocusChangeListener { hasFocus ->
            if (hasFocus) {
                refreshStatus()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        view.viewTreeObserver.addOnWindowFocusChangeListener(focusListener)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            view.viewTreeObserver.removeOnWindowFocusChangeListener(focusListener)
        }
    }

    val importOnnxLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri != null) {
            isImporting = true
            scope.launch {
                val ok = copyUriToFile(
                    context = context,
                    uri = uri,
                    dest = File(layaEngine.getModelsDir(), LayaEngine.DEFAULT_MODEL_FILENAME),
                )
                isImporting = false
                refreshStatus()
                Toast.makeText(
                    context,
                    if (ok) "Laya ONNX model imported." else "Failed to import ONNX file.",
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }
    }

    val importVocabLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri != null) {
            isImporting = true
            scope.launch {
                val ok = copyUriToFile(
                    context = context,
                    uri = uri,
                    dest = File(layaEngine.getModelsDir(), LayaEngine.DEFAULT_VOCAB_FILENAME),
                )
                isImporting = false
                refreshStatus()
                Toast.makeText(
                    context,
                    if (ok) "Tokenizer vocab.txt imported." else "Failed to import vocab file.",
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }
    }

    Scaffold(containerColor = Color.Black) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .padding(innerPadding)
                .verticalScroll(rememberScrollState()),
        ) {
            // Editorial Header
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 22.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Text(
                        text = "Unsend.",
                        fontFamily = FontFamily.Serif,
                        fontSize = 36.sp,
                        color = Color.White,
                    )
                    Text(
                        text = "ON-DEVICE",
                        fontSize = 10.sp,
                        letterSpacing = 2.sp,
                        color = Color(0xFF666666),
                    )
                }
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "MIRROR, NOT A WRITER",
                    fontSize = 11.sp,
                    letterSpacing = 2.2.sp,
                    color = Color(0xFF737373),
                )
            }

            EditorialHairline()

            // 01 Keyboard Service
            EditorialRow(
                index = "01",
                title = "Keyboard Service",
                status = if (isImeEnabled) "Enabled" else "Enable →",
                isActive = isImeEnabled,
                onClick = {
                    context.startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
                },
            )

            EditorialHairline(inset = true)

            // 02 Active Input
            EditorialRow(
                index = "02",
                title = "Active Input",
                status = if (isImeSelected) "Selected" else "Select →",
                isActive = isImeSelected,
                onClick = {
                    if (!isImeEnabled) {
                        Toast.makeText(context, "Enable Unsend Keyboard first.", Toast.LENGTH_SHORT).show()
                    } else {
                        val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
                        imm?.showInputMethodPicker()
                    }
                },
            )

            EditorialHairline(inset = true)

            // 03 Laya Engine
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 16.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Text(
                            text = "03",
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            color = Color(0xFF525252),
                        )
                        Text(
                            text = "Laya Engine",
                            fontSize = 15.sp,
                            color = Color(0xFFE5E5E5),
                        )
                    }
                    Text(
                        text = when (modelStatus) {
                            ModelStatus.READY_ONNX -> "ONNX Ready"
                            ModelStatus.PREVIEW_MODE -> "Built-in"
                            ModelStatus.NOT_CONFIGURED -> "Off"
                        },
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        color = if (modelStatus != ModelStatus.NOT_CONFIGURED) Color(0xFF22C55E) else Color(0xFFF59E0B),
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = buildString {
                        append(onnxFileName ?: "Built-in analyzer active out of the box")
                        if (hasVocab) append("  •  vocab.txt loaded")
                    },
                    fontSize = 12.sp,
                    color = Color(0xFF666666),
                    modifier = Modifier.padding(start = 30.dp),
                )

                if (isDownloading) {
                    Spacer(modifier = Modifier.height(10.dp))
                    LinearProgressIndicator(
                        progress = { downloadProgress },
                        color = Color.White,
                        trackColor = Color(0xFF1A1A1A),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 30.dp)
                            .height(2.dp),
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.padding(start = 30.dp),
                    horizontalArrangement = Arrangement.spacedBy(20.dp),
                ) {
                    Text(
                        text = when {
                            isDownloading -> "Downloading ${(downloadProgress * 100).toInt()}%…"
                            onnxFileName != null -> "Re-download ONNX →"
                            else -> "Download Laya ONNX →"
                        },
                        fontSize = 12.sp,
                        letterSpacing = 0.6.sp,
                        color = Color.White,
                        modifier = Modifier.clickable(enabled = !isDownloading && !isImporting) {
                            isDownloading = true
                            downloadProgress = 0f
                            scope.launch {
                                val result = layaEngine.downloadModel { p ->
                                    downloadProgress = p
                                }
                                isDownloading = false
                                refreshStatus()
                                Toast.makeText(
                                    context,
                                    if (result.isSuccess) {
                                        "Laya ONNX model downloaded."
                                    } else {
                                        "Download unavailable — built-in analyzer remains active."
                                    },
                                    Toast.LENGTH_SHORT,
                                ).show()
                            }
                        },
                    )
                    Text(
                        text = if (isImporting) "Importing…" else "Import .onnx →",
                        fontSize = 12.sp,
                        letterSpacing = 0.6.sp,
                        color = Color(0xFFA3A3A3),
                        modifier = Modifier.clickable(enabled = !isImporting && !isDownloading) {
                            importOnnxLauncher.launch(arrayOf("*/*"))
                        },
                    )
                }
            }

            EditorialHairline(inset = true)

            // 04 Preview Analyzer Toggle
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(
                        text = "04",
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF525252),
                    )
                    Column {
                        Text(
                            text = "Preview Analyzer",
                            fontSize = 15.sp,
                            color = Color(0xFFE5E5E5),
                        )
                        Text(
                            text = "Test live HUD & Reply mode without .onnx",
                            fontSize = 12.sp,
                            color = Color(0xFF666666),
                        )
                    }
                }
                Switch(
                    checked = previewEnabled,
                    onCheckedChange = { checked ->
                        layaEngine.setPreviewModeEnabled(checked)
                        refreshStatus()
                    },
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color(0xFF737373),
                        uncheckedTrackColor = Color(0xFF141414),
                        uncheckedBorderColor = Color(0xFF262626),
                    ),
                )
            }

            EditorialHairline()

            // Message Draft Sandbox
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF050505))
                    .padding(horizontal = 24.dp, vertical = 20.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "MESSAGE DRAFT",
                        fontSize = 10.sp,
                        letterSpacing = 2.sp,
                        color = Color(0xFF666666),
                    )
                    Text(
                        text = "COPY SAMPLE REPLY",
                        fontSize = 10.sp,
                        letterSpacing = 1.4.sp,
                        color = Color.White,
                        modifier = Modifier.clickable {
                            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                            val sample = "Hey, are we still on for the 3pm deadline today? Need an update."
                            clipboard?.setPrimaryClip(ClipData.newPlainText("Incoming", sample))
                            Toast.makeText(
                                context,
                                "Sample incoming message copied. Tap REPLY on the keyboard.",
                                Toast.LENGTH_SHORT,
                            ).show()
                        },
                    )
                }

                Spacer(modifier = Modifier.height(14.dp))

                BasicTextField(
                    value = sandboxDraft,
                    onValueChange = { sandboxDraft = it },
                    textStyle = TextStyle(
                        fontFamily = FontFamily.Serif,
                        fontSize = 19.sp,
                        lineHeight = 26.sp,
                        color = Color.White,
                    ),
                    cursorBrush = SolidColor(Color.White),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp),
                    decorationBox = { innerTextField ->
                        if (sandboxDraft.isEmpty()) {
                            Text(
                                text = "Tap here to write with Unsend…",
                                fontFamily = FontFamily.Serif,
                                fontSize = 19.sp,
                                color = Color(0xFF404040),
                            )
                        }
                        innerTextField()
                    },
                )
            }

            EditorialHairline()
        }
    }
}

@Composable
private fun EditorialHairline(inset: Boolean = false) {
    HorizontalDivider(
        modifier = if (inset) Modifier.padding(horizontal = 24.dp) else Modifier,
        thickness = 1.dp,
        color = Color(0xFF1A1A1A),
    )
}

@Composable
private fun EditorialRow(
    index: String,
    title: String,
    status: String,
    isActive: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(
                text = index,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
                color = Color(0xFF525252),
            )
            Text(
                text = title,
                fontSize = 15.sp,
                color = Color(0xFFE5E5E5),
            )
        }
        Text(
            text = status,
            fontSize = 13.sp,
            fontWeight = if (isActive) FontWeight.Normal else FontWeight.Medium,
            color = if (isActive) Color(0xFF22C55E) else Color.White,
        )
    }
}

private fun checkImeEnabled(context: Context): Boolean {
    val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager ?: return false
    val expectedId = ComponentName(context, UnsendImeService::class.java).flattenToShortString()
    return imm.enabledInputMethodList.any {
        it.packageName == context.packageName || it.id == expectedId
    }
}

private fun checkImeSelected(context: Context): Boolean {
    val current = Settings.Secure.getString(
        context.contentResolver,
        Settings.Secure.DEFAULT_INPUT_METHOD,
    ) ?: return false
    val expected = ComponentName(context, UnsendImeService::class.java).flattenToShortString()
    return current == expected || current.startsWith(context.packageName)
}

private suspend fun copyUriToFile(
    context: Context,
    uri: Uri,
    dest: File,
): Boolean = withContext(Dispatchers.IO) {
    runCatching {
        context.contentResolver.openInputStream(uri)?.use { input ->
            dest.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: return@withContext false
        dest.exists() && dest.length() > 0L
    }.getOrDefault(false)
}
