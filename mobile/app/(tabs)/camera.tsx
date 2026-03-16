import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Image as RNImage,
  Modal,
} from "react-native";
import { useState, useEffect, useRef } from "react";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";

const API_BASE = process.env.EXPO_PUBLIC_API_URL;

type Mode = "letter" | "word";

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState<"front" | "back">(
    "front",
  );
  const [mode, setMode] = useState<Mode>("letter");

  const [prediction, setPrediction] = useState<{
    letter: string;
    confidence: number;
  } | null>(null);
  const [isActive, setIsActive] = useState(true);

  const [isRecording, setIsRecording] = useState(false);
  const [wordPrediction, setWordPrediction] = useState<{
    word: string;
    confidence: number;
  } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  const [outputText, setOutputText] = useState("");
  const [debugImage, setDebugImage] = useState<string | null>(null);
  const [isDebugging, setIsDebugging] = useState(false);

  const device = useCameraDevice(cameraPosition);
  const cameraRef = useRef<Camera>(null);

  // Ref to abort in-flight letter prediction requests when a new one fires
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    requestPermission();
  }, []);

  // ── Letter mode: continuous prediction every 500ms ─────────────────────────
  useEffect(() => {
    if (!hasPermission || mode !== "letter") return;

    const interval = setInterval(async () => {
      if (!cameraRef.current || !isActive) return;

      // Cancel any in-flight request before firing a new one
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      try {
        const snapshot = await cameraRef.current.takePhoto({
          enableShutterSound: false,
        });
        const form = new FormData();
        form.append("file", {
          uri: `file://${snapshot.path}`,
          type: "image/jpeg",
          name: "frame.jpg",
        } as any);

        const res = await fetch(`${API_BASE}/predict/letter`, {
          method: "POST",
          body: form,
          signal: abortRef.current.signal,
        });
        const data = await res.json();
        if (data.letter && data.confidence > 0.7) {
          setPrediction({ letter: data.letter, confidence: data.confidence });
        }
      } catch (e: any) {
        // Ignore abort errors — these are intentional
        if (e?.name !== "AbortError") {
          console.error("Letter prediction error:", e);
        }
      }
    }, 500);

    return () => {
      clearInterval(interval);
      abortRef.current?.abort();
    };
  }, [hasPermission, isActive, mode]);

  // ── Countdown helper — uses an interval so state updates render correctly ──
  const startCountdown = (): Promise<void> => {
    return new Promise((resolve) => {
      let count = 3;
      setCountdown(count);
      const interval = setInterval(() => {
        count -= 1;
        if (count === 0) {
          clearInterval(interval);
          setCountdown(null);
          resolve();
        } else {
          setCountdown(count);
        }
      }, 1000);
    });
  };

  const handleDebug = async () => {
    if (!cameraRef.current || isRecording) return;
    setIsDebugging(true);
    try {
      await cameraRef.current.startRecording({
        fileType: "mp4",
        onRecordingFinished: async (video) => {
          try {
            const form = new FormData();
            form.append("file", {
              uri: `file://${video.path}`,
              type: "video/mp4",
              name: "debug.mp4",
            } as any);
            const res = await fetch(`${API_BASE}/debug/keypoints`, {
              method: "POST",
              body: form,
            });
            const data = await res.json();
            if (data.image)
              setDebugImage(`data:image/jpeg;base64,${data.image}`);
          } catch (e) {
            console.error("Debug error:", e);
          } finally {
            setIsDebugging(false);
          }
        },
        onRecordingError: () => setIsDebugging(false),
      });
      setTimeout(async () => {
        await cameraRef.current?.stopRecording();
      }, 1000);
    } catch (e) {
      console.error("Debug start error:", e);
      setIsDebugging(false);
    }
  };

  const handleRecordToggle = async () => {
    if (isRecording || isProcessing || countdown !== null) return;

    await startCountdown();

    if (!cameraRef.current) return;
    setIsRecording(true);
    setWordPrediction(null);

    try {
      await cameraRef.current.startRecording({
        fileType: "mp4",
        onRecordingFinished: async (video) => {
          setIsProcessing(true);
          try {
            const form = new FormData();
            form.append("file", {
              uri: `file://${video.path}`,
              type: "video/mp4",
              name: "sign.mp4",
            } as any);
            const res = await fetch(`${API_BASE}/predict/word`, {
              method: "POST",
              body: form,
            });
            const data = await res.json();
            if (data.word) {
              setWordPrediction({
                word: data.word,
                confidence: data.confidence,
              });
            }
          } catch (e) {
            console.error("Word prediction error:", e);
          } finally {
            setIsProcessing(false);
          }
        },
        onRecordingError: (e) => {
          console.error("Recording error:", e);
          setIsRecording(false);
          setIsProcessing(false);
        },
      });

      setTimeout(async () => {
        setIsRecording(false);
        await cameraRef.current?.stopRecording();
      }, 2000);
    } catch (e) {
      console.error("Start recording error:", e);
      setIsRecording(false);
    }
  };

  const handleAddToOutput = () => {
    if (mode === "letter" && prediction?.letter) {
      setOutputText((prev) => prev + prediction.letter);
    } else if (mode === "word" && wordPrediction?.word) {
      setOutputText((prev) =>
        prev ? prev + " " + wordPrediction.word : wordPrediction.word,
      );
    }
  };

  const handleClear = () => {
    setOutputText("");
    setPrediction(null);
    setWordPrediction(null);
    setIsRecording(false);
    setIsProcessing(false);
  };

  const switchMode = (newMode: Mode) => {
    setMode(newMode);
    setPrediction(null);
    setWordPrediction(null);
    setIsRecording(false);
    setIsProcessing(false);
    setCountdown(null);
  };

  if (!hasPermission) {
    return (
      <View style={styles.centeredState}>
        <Text style={styles.stateTitle}>Camera Access Needed</Text>
        <Text style={styles.stateSubtitle}>
          OpenHands needs camera access to detect ASL gestures.
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
        >
          <Text style={styles.permissionButtonText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centeredState}>
        <Text style={styles.stateTitle}>No Camera Found</Text>
      </View>
    );
  }

  const confidencePct = prediction
    ? Math.round(prediction.confidence * 100)
    : 0;
  const wordConfidencePct = wordPrediction
    ? Math.round(wordPrediction.confidence * 100)
    : 0;

  const FRAME_TOP = "15%";
  const FRAME_BOTTOM = "15%";
  const FRAME_SIDE = "6%";
  const OVERLAY_COLOR = "rgba(10,10,20,0.72)";
  const borderColor = isRecording
    ? "rgba(255,69,58,0.85)"
    : "rgba(99,91,255,0.5)";

  return (
    <View style={styles.root}>
      <View style={styles.cameraContainer}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          video={mode === "word"}
          audio={false}
        />

        {/* Dark overlays outside the signing frame */}
        <View
          style={[
            styles.overlayTop,
            { height: FRAME_TOP, backgroundColor: OVERLAY_COLOR },
          ]}
        />
        <View
          style={[
            styles.overlayBottom,
            { height: FRAME_BOTTOM, backgroundColor: OVERLAY_COLOR },
          ]}
        />
        <View
          style={[
            styles.overlayLeft,
            {
              top: FRAME_TOP,
              bottom: FRAME_BOTTOM,
              width: FRAME_SIDE,
              backgroundColor: OVERLAY_COLOR,
            },
          ]}
        />
        <View
          style={[
            styles.overlayRight,
            {
              top: FRAME_TOP,
              bottom: FRAME_BOTTOM,
              width: FRAME_SIDE,
              backgroundColor: OVERLAY_COLOR,
            },
          ]}
        />

        {/* Frame border */}
        <View
          style={[
            styles.frameBox,
            {
              top: FRAME_TOP,
              left: FRAME_SIDE,
              right: FRAME_SIDE,
              bottom: FRAME_BOTTOM,
              borderColor,
            },
          ]}
        />

        {/* Top bar */}
        <View style={styles.cameraTopRow}>
          <View style={styles.livePill}>
            <View
              style={[styles.liveDot, isRecording && styles.liveDotRecording]}
            />
            <Text style={styles.liveText}>
              {isRecording
                ? "REC"
                : mode === "letter"
                  ? isActive
                    ? "LIVE"
                    : "PAUSED"
                  : "READY"}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.flipButton}
            onPress={() =>
              setCameraPosition((p) => (p === "front" ? "back" : "front"))
            }
          >
            <Text style={styles.flipText}>⟳</Text>
          </TouchableOpacity>
        </View>

        {/* Center label */}
        <View style={styles.frameLabelWrap} pointerEvents="none">
          {mode === "word" &&
            (countdown !== null ? (
              <Text style={styles.countdownText}>{countdown}</Text>
            ) : (
              <Text style={styles.frameLabelText}>
                {isRecording ? "Signing..." : "Position your upper body here"}
              </Text>
            ))}
        </View>
      </View>

      {/* Bottom panel */}
      <View style={styles.panel}>
        <View style={styles.modeToggle}>
          <TouchableOpacity
            style={[styles.modeBtn, mode === "letter" && styles.modeBtnActive]}
            onPress={() => switchMode("letter")}
          >
            <Text
              style={[
                styles.modeBtnText,
                mode === "letter" && styles.modeBtnTextActive,
              ]}
            >
              Letter
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.modeBtn, mode === "word" && styles.modeBtnActive]}
            onPress={() => switchMode("word")}
          >
            <Text
              style={[
                styles.modeBtnText,
                mode === "word" && styles.modeBtnTextActive,
              ]}
            >
              Word
            </Text>
          </TouchableOpacity>
        </View>

        {mode === "letter" ? (
          <View style={styles.detectionCard}>
            <View>
              <Text style={styles.detectedLabel}>Detected</Text>
              <Text style={styles.bigLetter}>{prediction?.letter ?? "—"}</Text>
            </View>
            <View style={styles.detectionRight}>
              <View style={styles.confBadge}>
                <Text style={styles.confNumber}>
                  {prediction ? `${confidencePct}` : "—"}
                </Text>
                {prediction && <Text style={styles.confUnit}>%</Text>}
              </View>
              <Text style={styles.confLabel}>confidence</Text>
            </View>
          </View>
        ) : (
          <View style={styles.detectionCard}>
            <View>
              <Text style={styles.detectedLabel}>
                {isProcessing
                  ? "Processing..."
                  : isRecording
                    ? "Recording..."
                    : "Detected"}
              </Text>
              <Text style={[styles.bigLetter, styles.bigWord]}>
                {isProcessing ? "..." : (wordPrediction?.word ?? "—")}
              </Text>
            </View>
            {wordPrediction && !isProcessing && (
              <View style={styles.detectionRight}>
                <View style={styles.confBadge}>
                  <Text style={styles.confNumber}>{wordConfidencePct}</Text>
                  <Text style={styles.confUnit}>%</Text>
                </View>
                <Text style={styles.confLabel}>confidence</Text>
              </View>
            )}
          </View>
        )}

        <View style={styles.divider} />

        <View style={styles.outputRow}>
          <View style={styles.outputTextWrap}>
            <Text style={styles.outputLabel}>Output</Text>
            <Text style={styles.outputWord} numberOfLines={1}>
              {outputText || "..."}
            </Text>
          </View>
          <View style={styles.outputActions}>
            <TouchableOpacity
              style={styles.addButton}
              onPress={handleAddToOutput}
            >
              <Text style={styles.addButtonText}>+ Add</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.clearButton} onPress={handleClear}>
              <Text style={styles.clearButtonText}>Clear</Text>
            </TouchableOpacity>
          </View>
        </View>

        {mode === "letter" ? (
          <TouchableOpacity
            style={[styles.pauseButton, !isActive && styles.resumeButton]}
            onPress={() => setIsActive((p) => !p)}
          >
            <Text style={styles.pauseButtonText}>
              {isActive ? "Pause Detection" : "Resume Detection"}
            </Text>
          </TouchableOpacity>
        ) : (
          <>
            <TouchableOpacity
              style={[
                styles.recordButton,
                isRecording && styles.recordButtonActive,
              ]}
              onPress={handleRecordToggle}
              activeOpacity={0.8}
            >
              <Text style={styles.recordButtonText}>
                {isProcessing
                  ? "Processing..."
                  : isRecording
                    ? "Recording..."
                    : countdown !== null
                      ? `Starting in ${countdown}...`
                      : "Tap to Sign a Word"}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.debugButton}
              onPress={handleDebug}
              disabled={isDebugging}
            >
              <Text style={styles.debugButtonText}>
                {isDebugging ? "Detecting..." : "Debug Keypoints"}
              </Text>
            </TouchableOpacity>

            <Modal visible={!!debugImage} transparent animationType="fade">
              <View style={styles.modalOverlay}>
                <View style={styles.modalContent}>
                  <Text style={styles.modalTitle}>Keypoint Detection</Text>
                  {debugImage && (
                    <RNImage
                      source={{ uri: debugImage }}
                      style={styles.debugImageView}
                      resizeMode="contain"
                    />
                  )}
                  <TouchableOpacity
                    style={styles.modalClose}
                    onPress={() => setDebugImage(null)}
                  >
                    <Text style={styles.modalCloseText}>Close</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </Modal>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0a0a14" },
  cameraContainer: { flex: 1, position: "relative", overflow: "hidden" },
  overlayTop: { position: "absolute", top: 0, left: 0, right: 0 },
  overlayBottom: { position: "absolute", bottom: 0, left: 0, right: 0 },
  overlayLeft: { position: "absolute", left: 0 },
  overlayRight: { position: "absolute", right: 0 },
  frameBox: { position: "absolute", borderWidth: 2, borderRadius: 0 },
  cameraTopRow: {
    position: "absolute",
    top: 60,
    left: 16,
    right: 16,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    zIndex: 10,
  },
  livePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(10,10,20,0.7)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  liveDotRecording: {
    backgroundColor: "#ff453a",
    shadowColor: "#ff453a",
    shadowOpacity: 1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 0 },
  },
  liveText: {
    fontSize: 11,
    fontWeight: "600",
    color: "rgba(255,255,255,0.7)",
    letterSpacing: 0.8,
  },
  flipButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "rgba(10,10,20,0.7)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  flipText: { fontSize: 18, color: "#fff" },
  frameLabelWrap: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  countdownText: {
    fontSize: 64,
    fontWeight: "800",
    color: "#fff",
    textShadowColor: "rgba(99,91,255,0.8)",
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 20,
  },
  frameLabelText: {
    fontSize: 12,
    fontWeight: "600",
    color: "rgba(255,255,255,0.5)",
    backgroundColor: "rgba(10,10,20,0.6)",
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 10,
    overflow: "hidden",
  },
  panel: {
    backgroundColor: "#0a0a14",
    borderTopWidth: 1,
    borderTopColor: "rgba(255,255,255,0.06)",
    padding: 20,
    paddingBottom: 36,
  },
  modeToggle: {
    flexDirection: "row",
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    borderRadius: 12,
    padding: 3,
    marginBottom: 16,
  },
  modeBtn: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 9,
    alignItems: "center",
  },
  modeBtnActive: { backgroundColor: "#635bff" },
  modeBtnText: {
    fontSize: 13,
    fontWeight: "600",
    color: "rgba(255,255,255,0.3)",
  },
  modeBtnTextActive: { color: "#fff" },
  detectionCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  detectedLabel: {
    fontSize: 11,
    fontWeight: "500",
    color: "rgba(255,255,255,0.3)",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 2,
  },
  bigLetter: {
    fontSize: 72,
    fontWeight: "800",
    color: "#fff",
    lineHeight: 72,
    letterSpacing: -2,
  },
  bigWord: { fontSize: 36, lineHeight: 44, letterSpacing: -1 },
  detectionRight: { alignItems: "flex-end", paddingTop: 4 },
  confBadge: {
    flexDirection: "row",
    alignItems: "baseline",
    backgroundColor: "rgba(99,91,255,0.15)",
    borderWidth: 1,
    borderColor: "rgba(99,91,255,0.25)",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    gap: 2,
    marginBottom: 6,
  },
  confNumber: {
    fontSize: 28,
    fontWeight: "800",
    color: "#a89fff",
    letterSpacing: -1,
  },
  confUnit: { fontSize: 14, fontWeight: "600", color: "rgba(168,159,255,0.6)" },
  confLabel: {
    fontSize: 11,
    color: "rgba(255,255,255,0.25)",
    fontWeight: "500",
  },
  divider: {
    height: 1,
    backgroundColor: "rgba(255,255,255,0.06)",
    marginBottom: 16,
  },
  outputRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  outputTextWrap: { flex: 1, marginRight: 12 },
  outputLabel: {
    fontSize: 11,
    fontWeight: "500",
    color: "rgba(255,255,255,0.25)",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  outputWord: {
    fontSize: 20,
    fontWeight: "700",
    color: "rgba(255,255,255,0.6)",
    letterSpacing: 2,
  },
  outputActions: { flexDirection: "row", gap: 8 },
  addButton: {
    backgroundColor: "rgba(99,91,255,0.15)",
    borderWidth: 1,
    borderColor: "rgba(99,91,255,0.3)",
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  addButtonText: { fontSize: 13, fontWeight: "600", color: "#a89fff" },
  clearButton: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  clearButtonText: {
    fontSize: 13,
    fontWeight: "600",
    color: "rgba(255,255,255,0.3)",
  },
  pauseButton: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
  },
  resumeButton: {
    backgroundColor: "rgba(99,91,255,0.12)",
    borderColor: "rgba(99,91,255,0.25)",
  },
  pauseButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "rgba(255,255,255,0.4)",
  },
  recordButton: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
  },
  recordButtonActive: {
    backgroundColor: "rgba(255,69,58,0.15)",
    borderColor: "rgba(255,69,58,0.4)",
  },
  recordButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "rgba(255,255,255,0.4)",
  },
  debugButton: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    borderRadius: 14,
    paddingVertical: 10,
    alignItems: "center",
    marginTop: 8,
  },
  debugButtonText: {
    fontSize: 13,
    fontWeight: "600",
    color: "rgba(255,255,255,0.25)",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.85)",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  modalContent: {
    backgroundColor: "#0a0a14",
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.07)",
    padding: 20,
    width: "100%",
    alignItems: "center",
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 16,
  },
  debugImageView: {
    width: "100%",
    height: 400,
    borderRadius: 12,
    marginBottom: 16,
  },
  modalClose: {
    backgroundColor: "#635bff",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 32,
  },
  modalCloseText: { fontSize: 14, fontWeight: "700", color: "#fff" },
  centeredState: {
    flex: 1,
    backgroundColor: "#0a0a14",
    alignItems: "center",
    justifyContent: "center",
    padding: 40,
  },
  stateTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: "#fff",
    letterSpacing: -0.5,
    marginBottom: 10,
    textAlign: "center",
  },
  stateSubtitle: {
    fontSize: 15,
    color: "rgba(255,255,255,0.3)",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 28,
  },
  permissionButton: {
    backgroundColor: "#635bff",
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 28,
  },
  permissionButtonText: { fontSize: 15, fontWeight: "700", color: "#fff" },
});
