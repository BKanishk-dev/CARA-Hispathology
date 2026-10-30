// Plain-language explanations generated from the numbers returned by /api/compare.

const P = (v) => `${(v * 100).toFixed(1)}%`;

const level = (v, lo, hi, words) => (v >= hi ? words[2] : v >= lo ? words[1] : words[0]);

export function explainImage(result) {
  const { quality, blur, stain } = result.image;
  const parts = [];
  parts.push(
    `Overall image quality Q is ${P(quality)}, which is ${level(quality, 0.45, 0.7, ["low", "moderate", "good"])}.`,
  );
  parts.push(
    `Sharpness is ${P(blur)} (${level(blur, 0.4, 0.7, [
      "the image looks blurry, so fine nuclear detail is lost",
      "slightly soft but usable",
      "the image is sharp and in focus",
    ])}).`,
  );
  parts.push(
    `Staining is ${P(stain)} (${level(stain, 0.4, 0.7, [
      "the H&E colours are faint or uneven, so the tissue is hard to read",
      "the H&E staining is acceptable",
      "clear pink (eosin) and purple (haematoxylin) staining is present",
    ])}).`,
  );
  if (quality < 0.45) {
    parts.push("Because quality is low, every model's CARA score is pulled down and more predictions will be flagged Unreliable.");
  }
  return parts.join(" ");
}

export function explainConsensus(result) {
  const { label, votes, total } = result.consensus;
  const reliable = result.models.filter((m) => m.reliable);
  const agreeWord = votes === total ? "all" : votes >= total / 2 ? "most" : "only some";
  let s = `${agreeWord === "all" ? "All" : agreeWord === "most" ? "Most" : "Only some"} models (${votes} of ${total}) vote for "${label}". `;
  s += `${reliable.length} of ${total} models pass the CARA reliability check. `;
  s += reliable.length === 0
    ? "No model is trusted for this image, so it should go to a pathologist."
    : votes === total
      ? "Strong agreement across different architectures makes this result more dependable."
      : "The models disagree, so treat the answer with caution and prefer the reliable ones.";
  return s;
}

export function explainModel(m, consensusLabel) {
  const s = [];
  s.push(
    `${m.name} predicts "${m.predicted_label}"` +
      (m.top3?.[1] ? `, with "${m.top3[1].label}" (${P(m.top3[1].prob)}) as the next most likely class.` : "."),
  );
  s.push(
    `Softmax confidence is ${P(m.confidence)}: the model is ${level(m.confidence, 0.5, 0.8, ["unsure", "fairly sure", "very sure"])} of its own answer.`,
  );
  s.push(
    `Image quality is ${P(m.quality)} (${level(m.quality, 0.45, 0.7, ["poor", "fair", "good"])}).`,
  );
  s.push(
    `Class reliability is ${P(m.class_reliability)}: on validation images this model was right ${P(m.class_reliability)} of the time when it predicted this class.`,
  );
  s.push(
    `These combine into a CARA score of ${P(m.cara_score)} against the threshold of ${P(m.threshold)} for this class, ` +
      (m.reliable
        ? "so the prediction is marked Reliable and can be used."
        : "so it falls short and is marked Unreliable; an expert should review it."),
  );
  if (consensusLabel) {
    s.push(
      m.predicted_label === consensusLabel
        ? "It agrees with the majority of models."
        : `It disagrees with the majority, which chose "${consensusLabel}".`,
    );
  }
  return s.join(" ");
}
