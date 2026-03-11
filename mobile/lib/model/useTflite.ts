import { useTensorflowModel } from "react-native-fast-tflite";
import labels from "@/assets/model/labels.json";

export function useASLModel() {
  const model = useTensorflowModel(
    require("../../assets/model/asl_hand_model_v3.tflite"),
  );

  const predict = (
    input: Float32Array,
  ): { letter: string; confidence: number } | null => {
    if (model.state !== "loaded" || !model.model) return null;

    const outputs = model.model.runSync([input]);
    const scores = outputs[0] as Float32Array;

    let maxIdx = 0;
    let maxScore = scores[0];
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIdx = i;
      }
    }

    const letter = (labels as Record<string, string>)[String(maxIdx)];
    return { letter, confidence: maxScore };
  };

  return { predict, isReady: model.state === "loaded" };
}
