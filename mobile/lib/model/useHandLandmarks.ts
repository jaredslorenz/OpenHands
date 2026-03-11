import { useFrameProcessor } from "react-native-vision-camera";
import { runOnJS } from "react-native-reanimated";
import { useSharedValue } from "react-native-reanimated";

const API_URL = process.env.EXPO_PUBLIC_API_URL + "/predict";

export function useHandLandmarks(
  onPrediction: (letter: string, confidence: number) => void,
) {
  const lastSent = useSharedValue(0);

  const sendFrame = async (arrayBuffer: ArrayBuffer) => {
    try {
      const uint8 = new Uint8Array(arrayBuffer);
      const blob = new Blob([uint8], { type: "image/jpeg" });
      const form = new FormData();
      form.append("file", blob as any, "frame.jpg");

      const res = await fetch(API_URL, { method: "POST", body: form });
      const data = await res.json();
      if (data.letter) onPrediction(data.letter, data.confidence);
    } catch (e) {
      console.error("API error:", e);
    }
  };

  const frameProcessor = useFrameProcessor((frame) => {
    "worklet";
    const now = Date.now();
    if (now - lastSent.value < 500) return;
    lastSent.value = now;

    const buffer = frame.toArrayBuffer();
    runOnJS(sendFrame)(buffer);
  }, []);

  return { frameProcessor };
}
