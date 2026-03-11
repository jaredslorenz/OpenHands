import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Dimensions,
} from "react-native";
import { useRouter } from "expo-router";

const { width } = Dimensions.get("window");

const COLORS = {
  bg: "#0a0a14",
  card: "rgba(255,255,255,0.04)",
  cardBorder: "rgba(255,255,255,0.07)",
  accent: "#635bff",
  accentSoft: "rgba(99,91,255,0.15)",
  accentBorder: "rgba(99,91,255,0.25)",
  accentText: "#a89fff",
  white: "#ffffff",
  muted: "rgba(255,255,255,0.3)",
};

const ASL_ALPHABET = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  "del",
  "_",
  "nun",
];

export default function HomeScreen() {
  const router = useRouter();

  return (
    <View style={styles.root}>
      <View style={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.statusPill}>
            <View style={styles.statusDot} />
            <Text style={styles.statusText}>Ready</Text>
          </View>
          <Text style={styles.appName}>OpenHands</Text>
          <Text style={styles.appSubtitle}>
            Real-time ASL to text translation powered by AI
          </Text>
        </View>

        {/* CTA Button */}
        <TouchableOpacity
          style={styles.ctaButton}
          onPress={() => router.push("/(tabs)/camera")}
          activeOpacity={0.85}
        >
          <Text style={styles.ctaText}>Start Translating</Text>
          <Text style={styles.ctaArrow}>→</Text>
        </TouchableOpacity>

        {/* Info Cards */}
        <View style={styles.cardsRow}>
          <View style={styles.infoCard}>
            <Text style={styles.infoCardNumber}>100</Text>
            <Text style={styles.infoCardLabel}>Words supported</Text>
          </View>
          <View style={styles.infoCard}>
            <Text style={styles.infoCardNumber}>Live</Text>
            <Text style={styles.infoCardLabel}>Real-time detection</Text>
          </View>
        </View>

        {/* ASL Reference */}
        <Text style={styles.sectionLabel}>ASL ALPHABET</Text>
        <View style={styles.alphabetGrid}>
          {ASL_ALPHABET.map((letter) => (
            <View key={letter} style={styles.letterCard}>
              <Text style={styles.letterCardLetter}>{letter}</Text>
            </View>
          ))}
        </View>

        {/* Bottom Note */}
        <View style={styles.bottomNote}>
          <Text style={styles.bottomNoteText}>
            Position your hand clearly in frame for best results. Ensure good
            lighting.
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#0a0a14",
  },
  scrollContent: {
    flex: 1,
    paddingTop: 60,
    paddingBottom: 16,
    paddingHorizontal: 20,
    justifyContent: "space-between",
  },

  header: {
    marginBottom: 16,
  },
  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: COLORS.accentSoft,
    borderWidth: 1,
    borderColor: COLORS.accentBorder,
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 5,
    alignSelf: "flex-start",
    marginBottom: 12,
  },
  statusDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: COLORS.accent,
    shadowColor: COLORS.accent,
    shadowOpacity: 1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 0 },
  },
  statusText: {
    fontSize: 11,
    fontWeight: "600",
    color: COLORS.accentText,
    letterSpacing: 0.3,
  },
  appName: {
    fontSize: 32,
    fontWeight: "800",
    color: COLORS.white,
    letterSpacing: -1,
    marginBottom: 6,
  },
  appSubtitle: {
    fontSize: 14,
    color: COLORS.muted,
    lineHeight: 20,
    fontWeight: "400",
    maxWidth: 280,
  },

  ctaButton: {
    backgroundColor: COLORS.accent,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 24,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
    shadowColor: COLORS.accent,
    shadowOpacity: 0.35,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 8 },
  },
  ctaText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#fff",
    letterSpacing: -0.3,
  },
  ctaArrow: {
    fontSize: 18,
    color: "rgba(255,255,255,0.7)",
  },

  cardsRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 20,
  },
  infoCard: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 16,
    padding: 14,
  },
  infoCardNumber: {
    fontSize: 22,
    fontWeight: "800",
    color: COLORS.white,
    letterSpacing: -0.5,
    marginBottom: 2,
  },
  infoCardLabel: {
    fontSize: 11,
    color: COLORS.muted,
    fontWeight: "500",
  },

  sectionLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: COLORS.muted,
    letterSpacing: 0.12,
    textTransform: "uppercase",
    marginBottom: 10,
  },

  alphabetGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginBottom: 16,
  },
  letterCard: {
    width: (width - 40 - 6 * 6) / 7,
    aspectRatio: 1,
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  letterCardLetter: {
    fontSize: 13,
    fontWeight: "700",
    color: COLORS.white,
    textAlign: "center",
    paddingBottom: 10,
  },

  bottomNote: {
    backgroundColor: COLORS.accentSoft,
    borderWidth: 1,
    borderColor: COLORS.accentBorder,
    borderRadius: 14,
    padding: 12,
  },
  bottomNoteText: {
    fontSize: 12,
    color: COLORS.accentText,
    lineHeight: 18,
    fontWeight: "400",
  },
});
