const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Yohannes Zewde";
pres.title = "Amharic-English NMT — Implementation Phases 1-3";
pres.subject = "Seq2Seq Transformer from Scratch";

// ─── THEME (extracted from existing deck) ───────────────────────────────────
const T = {
  accent1: "4472C4",  // blue
  accent2: "ED7D31",  // orange
  accent3: "A5A5A5",  // gray
  accent4: "FFC000",  // yellow
  accent5: "5B9BD5",  // light blue
  accent6: "70AD47",  // green
  dk2:     "44546A",  // dark blue-gray
  lt2:     "E7E6E6",  // light gray bg
  white:   "FFFFFF",
  black:   "000000",
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
const FONT_TITLE = "Calibri Light";
const FONT_BODY  = "Calibri";

function topBar(slide) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.032, fill: { color: T.accent1 }, line: { color: T.accent1 } });
}

function footerBar(slide) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.594, w: 10, h: 0.032, fill: { color: T.accent1 }, line: { color: T.accent1 } });
}

function sectionLabel(slide, text, x = 0.6, y = 0.343) {
  slide.addText(text, { x, y, w: 4, h: 0.21, fontSize: 10, color: T.accent1, bold: true, fontFace: FONT_BODY });
}

function slideTitle(slide, text, x = 0.6, y = 0.616, w = 9, h = 0.54) {
  slide.addText(text, { x, y, w, h, fontSize: 28, color: T.dk2, bold: true, fontFace: FONT_TITLE, margin: 0 });
}

function card(slide, x, y, w, h, fillColor) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fillColor || T.lt2 }, line: { color: T.lt2 } });
}

function pill(slide, x, y, size, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color }, line: { color } });
}

function pageNum(slide, n) {
  slide.addText(String(n), { x: 9.3, y: 5.12, w: 0.4, h: 0.28, fontSize: 10, color: T.accent3, fontFace: FONT_BODY, align: "right" });
}


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — COVER
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);

  // Decorative line 1
  slide.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 1.395, w: 0.5, h: 0.016, fill: { color: T.accent1 }, line: { color: T.accent1 } });
  // Decorative line 2
  slide.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 3.634, w: 0.5, h: 0.011, fill: { color: T.accent1 }, line: { color: T.accent1 } });
  // Decorative line 3
  slide.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 4.127, w: 0.5, h: 0.016, fill: { color: T.accent1 }, line: { color: T.accent1 } });

  slide.addText("Breaking Language Barriers", { x: 0.704, y: 1.544, w: 8.6, h: 0.672, fontSize: 36, color: T.dk2, bold: true, fontFace: FONT_TITLE, align: "center" });
  slide.addText("Amharic\u2013English Neural Machine Translation", { x: 1.085, y: 2.303, w: 7.83, h: 0.36, fontSize: 20, color: T.accent1, bold: false, fontFace: FONT_BODY, align: "center" });
  slide.addText("Implementation Phases 1\u20133 Complete", { x: 2.694, y: 2.856, w: 4.61, h: 0.282, fontSize: 14, color: T.accent2, bold: true, fontFace: FONT_BODY, align: "center" });

  // Amharic greeting
  slide.addText("\u1200\u12CB\u12CB\u121B", { x: 3.697, y: 3.387, w: 0.977, h: 0.495, fontSize: 28, color: T.dk2, bold: true, fontFace: "Nyala", align: "center" });
  slide.addText("Hello", { x: 5.099, y: 3.387, w: 1.234, h: 0.495, fontSize: 28, color: T.accent1, bold: true, fontFace: FONT_TITLE, align: "center" });

  slide.addText("Yohannes Zewde \u00b7 IT Student & ML Intern", { x: 2.958, y: 4.289, w: 4.085, h: 0.282, fontSize: 11, color: T.dk2, fontFace: FONT_BODY, align: "center" });
  footerBar(slide);
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — (blank section divider — keep as is)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — SECTION 1: INTRODUCTION
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 1 \u2014 INTRODUCTION");
  slideTitle(slide, "Demystifying ML and NLP");
  footerBar(slide);
  pageNum(slide, 3);

  // ML card
  card(slide, 0.601, 1.403, 4.581, 3.945);
  pill(slide, 0.906, 1.606, 0.349, T.accent1);
  slide.addText("ML", { x: 0.968, y: 1.665, w: 0.353, h: 0.224, fontSize: 10, color: T.white, bold: true, fontFace: FONT_BODY });
  slide.addText("Machine Learning", { x: 0.905, y: 2.106, w: 2.692, h: 0.308, fontSize: 13, color: T.dk2, bold: true, fontFace: FONT_TITLE });
  slide.addText([
    { text: '"If word X, then word Y"\u2014replaced by', options: { breakLine: true } },
    { text: "learned patterns from data." }
  ], { x: 0.905, y: 2.531, w: 3.852, h: 0.897, fontSize: 11, color: T.dk2, fontFace: FONT_BODY });

  // ML bottom bar
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.906, y: 3.597, w: 3.589, h: 0.719, fill: { color: T.accent1 }, line: { color: T.accent1 } });
  slide.addText("Traditional programming tells the computer exactly what to do. Machine Learning takes a different approach\u2014we provide the model with thousands of examples and let it discover the patterns on its own.", {
    x: 0.905, y: 3.720, w: 3.853, h: 0.582, fontSize: 9, color: T.white, fontFace: FONT_BODY
  });

  // NLP card
  card(slide, 5.201, 1.403, 4.573, 3.942);
  pill(slide, 5.506, 1.606, 0.349, T.accent2);
  slide.addText("NLP", { x: 5.541, y: 1.673, w: 0.424, h: 0.208, fontSize: 10, color: T.white, bold: true, fontFace: FONT_BODY });
  slide.addText("Natural Language Processing", { x: 5.505, y: 2.106, w: 4.4, h: 0.308, fontSize: 13, color: T.dk2, bold: true, fontFace: FONT_TITLE });
  slide.addText("A branch of AI that gives computers the ability to understand text and spoken words in much the same way human beings can.", {
    x: 5.505, y: 2.531, w: 4.277, h: 1.257, fontSize: 11, color: T.dk2, fontFace: FONT_BODY
  });

  // NLP bottom bar
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.506, y: 3.922, w: 3.589, h: 0.719, fill: { color: T.accent2 }, line: { color: T.accent2 } });
  slide.addText("Reading, summarizing, translating\u2014all powered by NLP.", { x: 5.505, y: 4.045, w: 4.341, h: 0.582, fontSize: 9, color: T.white, fontFace: FONT_BODY });

  // Transformers card (new)
  card(slide, 0.601, 1.403, 4.581, 3.945);
  pill(slide, 0.906, 1.606, 0.349, T.accent6);
  slide.addText("TF", { x: 0.955, y: 1.673, w: 0.25, h: 0.208, fontSize: 10, color: T.white, bold: true, fontFace: FONT_BODY });
  slide.addText("Seq2Seq Transformer", { x: 1.30, y: 1.665, w: 2.5, h: 0.224, fontSize: 11, color: T.dk2, bold: true, fontFace: FONT_TITLE });
  slide.addText("Built from Scratch in PyTorch", { x: 0.905, y: 2.106, w: 3.5, h: 0.308, fontSize: 11, color: T.accent6, bold: false, fontFace: FONT_BODY });
  slide.addText("The modern architecture powering AI like ChatGPT and Google Translate. We built it ourselves using PyTorch\u2019s nn.Transformer, implementing positional encoding, multi-head attention, encoder-decoder layers, and beam search decoding from the ground up.", {
    x: 0.905, y: 2.531, w: 4.0, h: 1.5, fontSize: 9, color: T.dk2, fontFace: FONT_BODY
  });
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.506, y: 3.922, w: 3.589, h: 0.719, fill: { color: T.accent6 }, line: { color: T.accent6 } });
  slide.addText("74M parameters \u00b7 d_model=512 \u00b7 6 encoder + 6 decoder layers", { x: 5.505, y: 4.045, w: 4.341, h: 0.4, fontSize: 9, color: T.white, fontFace: FONT_BODY });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — KEY CONCEPTS
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 1 \u2014 KEY CONCEPTS");
  slideTitle(slide, "Translation Tech & Key Concepts");
  footerBar(slide);
  pageNum(slide, 4);

  const concepts = [
    {
      color: T.accent1,
      label: "Neural Machine Translation (NMT)",
      desc: "Using artificial, brain-like networks to translate text. It understands whole-sentence context\u2014not just word-by-word substitution. Modern NMT is powered by the Transformer architecture.",
    },
    {
      color: T.accent2,
      label: "Low-Resource Language",
      desc: "A language that, despite having millions of speakers (like Amharic with 30M+), has very little digitized, translated data available on the internet to train AI systems effectively.",
    },
    {
      color: T.accent6,
      label: "Parallel Corpus",
      desc: 'The ultimate "dictionary" for AI. A massive dataset where every Amharic sentence has an exact English equivalent next to it\u2014the foundation of our 380K+ pair collection.',
    },
  ];

  const cardW = 8.802;
  const cardH = 1.287;
  const cardX = 0.599;
  let cardY = 1.347;

  concepts.forEach((c, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: cardX, y: cardY, w: cardW, h: cardH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    // accent bar
    slide.addShape(pres.shapes.RECTANGLE, { x: cardX, y: cardY, w: 0.06, h: cardH, fill: { color: c.color }, line: { color: c.color } });
    slide.addText(c.label, { x: cardX + 0.15, y: cardY + 0.15, w: 8.5, h: 0.276, fontSize: 12, color: c.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(c.desc, { x: cardX + 0.15, y: cardY + 0.47, w: 8.5, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
    cardY += cardH + 0.141;
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — VOCABULARY (updated)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 1 \u2014 VOCABULARY");
  slideTitle(slide, "The Advanced Vocabulary (Simplified)");
  footerBar(slide);
  pageNum(slide, 5);

  const vocab = [
    {
      color: T.accent1,
      abbr: "TF",
      name: "TRANSFORMER",
      subtitle: "The \u201cEngine\u201d",
      desc: "The modern architecture powering AI like ChatGPT or Google Translate. We built a Seq2Seq Transformer from scratch in PyTorch\u2014no pretrained weights.",
    },
    {
      color: T.accent5,
      abbr: "BPE",
      name: "SENTENCEPIECE BPE",
      subtitle: "Subword Tokenization",
      desc: "Breaking words into smaller subword pieces. Handles Amharic\u2019s 200+ Ge\u2019ez script characters with character_coverage=0.9995 and a 32K shared vocabulary.",
    },
    {
      color: T.accent6,
      abbr: "NMT",
      name: "SEQ2SEQ NMT",
      subtitle: "From Scratch",
      desc: "Instead of fine-tuning a pretrained model, we implemented the full encoder-decoder architecture ourselves\u2014positional encoding, attention, FFN, beam search.",
    },
    {
      color: T.accent2,
      abbr: "BLEU",
      name: "BLEU / chrF",
      subtitle: "The \u201cGrade\u201d",
      desc: "BLEU measures n-gram overlap; chrF adds character-level evaluation\u2014critical for morphologically rich Amharic text.",
    },
  ];

  const positions = [
    { x: 0.513, y: 1.153, w: 4.696, h: 2.472 },
    { x: 5.296, y: 1.180, w: 4.089, h: 2.446 },
    { x: 0.513, y: 3.714, w: 4.688, h: 1.828 },
    { x: 5.296, y: 3.747, w: 4.094, h: 1.776 },
  ];

  vocab.forEach((v, i) => {
    const p = positions[i];
    slide.addShape(pres.shapes.RECTANGLE, { x: p.x, y: p.y, w: p.w, h: p.h, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    pill(slide, p.x + 0.343, p.y + 0.345, 0.35, v.color);
    slide.addText(v.abbr, { x: p.x + 0.38, y: p.y + 0.38, w: 0.28, h: 0.22, fontSize: 9, color: T.white, bold: true, fontFace: FONT_BODY });
    slide.addText(v.name, { x: p.x + 0.343, y: p.y + 0.345, w: p.w - 0.4, h: 0.22, fontSize: 10, color: v.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(v.subtitle, { x: p.x + 0.343, y: p.y + 0.78, w: p.w - 0.4, h: 0.308, fontSize: 11, color: T.dk2, bold: true, fontFace: FONT_TITLE });
    slide.addText(v.desc, { x: p.x + 0.343, y: p.y + 1.17, w: p.w - 0.5, h: p.h - 1.4, fontSize: 9, color: T.dk2, fontFace: FONT_BODY });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — PROJECT OVERVIEW (updated)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 2 \u2014 PROJECT OVERVIEW");
  slideTitle(slide, "The Big Picture");
  footerBar(slide);
  pageNum(slide, 6);

  const items = [
    {
      num: "1",
      color: T.accent1,
      title: "The Ultimate Goal",
      desc: "Build a highly accurate Amharic\u2013English translation model and package it as a REST API\u2014a bridge so other web and mobile apps can easily use our translator.",
    },
    {
      num: "2",
      color: T.accent2,
      title: "Our Approach: Seq2Seq Transformer from Scratch",
      desc: "As required by supervisors, we are building a complete Transformer architecture from scratch in PyTorch\u2014not fine-tuning a pretrained model. This gives us full control over the architecture, training, and optimization pipeline.",
    },
    {
      num: "3",
      color: T.accent6,
      title: "Why It Matters",
      desc: "Amharic has 30M+ speakers but is digitally underserved. Improving Amharic NMT removes educational and economic barriers for millions of Ethiopians.",
    },
  ];

  let cardY = 1.347;
  const cardH = 1.310;
  const gap = 0.118;

  items.forEach((item, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: cardY, w: 8.874, h: cardH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    pill(slide, 0.904, cardY + 0.162, 0.302, item.color);
    slide.addText(item.num, { x: 1.027, y: cardY + 0.206, w: 0.136, h: 0.208, fontSize: 12, color: T.white, bold: true, fontFace: FONT_BODY, align: "center" });
    slide.addText(item.title, { x: 1.405, y: cardY + 0.157, w: 7.5, h: 0.276, fontSize: 12, color: item.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(item.desc, { x: 1.405, y: cardY + 0.512, w: 7.9, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
    cardY += cardH + gap;
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — DATA COLLECTION MILESTONE
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 3 \u2014 DATA COLLECTION MILESTONE");
  slideTitle(slide, "Gathering the Building Blocks");
  footerBar(slide);
  pageNum(slide, 7);

  slide.addText("In Machine Learning, data is food. A model is only as smart as the examples it consumes.", {
    x: -0.041, y: 1.626, w: 9.56, h: 0.282, fontSize: 13, color: T.dk2, italic: true, fontFace: FONT_BODY, align: "center"
  });

  const stats = [
    { label: "STARTED WITH", value: "2,000", sub: "sentence pairs", color: T.accent1 },
    { label: "GROWTH", value: "19,000%", sub: "increase in dataset", color: T.accent2 },
    { label: "CURRENT DATASET", value: "380K+", sub: "pristine parallel pairs", color: T.accent6 },
  ];

  const cardW = 3.0;
  const cardX = [1.6, 3.55, 5.5];
  const cardY = 2.3;

  stats.forEach((s, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: cardX[i], y: cardY, w: cardW, h: 2.2, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addText(s.label, { x: cardX[i], y: cardY + 0.18, w: cardW, h: 0.224, fontSize: 9, color: s.color, bold: true, fontFace: FONT_BODY, align: "center" });
    slide.addText(s.value, { x: cardX[i], y: cardY + 0.43, w: cardW, h: 1.0, fontSize: 48, color: s.color, bold: true, fontFace: FONT_TITLE, align: "center" });
    slide.addText(s.sub, { x: cardX[i], y: cardY + 1.38, w: cardW, h: 0.282, fontSize: 10, color: T.dk2, fontFace: FONT_BODY, align: "center" });
  });

  // vertical divider lines
  slide.addShape(pres.shapes.LINE, { x: 4.65, y: cardY + 0.1, w: 0, h: 1.9, line: { color: T.accent3, width: 1, dashType: "dash" } });
  slide.addShape(pres.shapes.LINE, { x: 6.6, y: cardY + 0.1, w: 0, h: 1.9, line: { color: T.accent3, width: 1, dashType: "dash" } });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — DATA SOURCES
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 3 \u2014 DATA SOURCES");
  slideTitle(slide, "Where Did the Data Come From?");
  footerBar(slide);
  pageNum(slide, 8);

  const sources = [
    { icon: "\u25a0", color: T.accent1, title: "Religious Texts", desc: "Bible, JW300\u2014highly consistent, professionally translated." },
    { icon: "\u25a0", color: T.accent2, title: "Legal & Law Books", desc: "Strict, formal, and precise terminology." },
    { icon: "\u25a0", color: T.accent5, title: "Colloquial Datasets", desc: "Conversational text to teach the model how people actually speak." },
    { icon: "\u25a0", color: T.accent6, title: "Web-Scraped Data", desc: "OPUS / Common Crawl\u2014massive, varied internet text." },
    { icon: "\u25a0", color: T.accent4, title: "News Articles", desc: "Multilingual news for modern vocabulary & current events." },
  ];

  const positions = [
    { x: 0.599, y: 1.466 }, { x: 5.099, y: 1.466 },
    { x: 0.599, y: 2.973 }, { x: 5.099, y: 2.973 },
    { x: 0.599, y: 4.480 },
  ];

  sources.forEach((s, i) => {
    const p = positions[i];
    slide.addShape(pres.shapes.RECTANGLE, { x: p.x, y: p.y, w: 0.052, h: 0.052, fill: { color: s.color }, line: { color: s.color } });
    slide.addText(s.title, { x: p.x + 0.2, y: p.y - 0.109, w: 3.5, h: 0.276, fontSize: 11, color: s.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(s.desc, { x: p.x + 0.2, y: p.y + 0.202, w: 4.5, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — ETHIOPIAN CONTEXT
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 3 \u2014 ETHIOPIAN CONTEXT");
  slideTitle(slide, "Local & Creative Sources");
  footerBar(slide);
  pageNum(slide, 9);

  const sections = [
    {
      title: "Corporate Digital Services & Apps",
      desc: "Scraping bilingual app localizations, user manuals, and Terms of Service from major local entities\u2014Ethio Telecom / Telebirr, Commercial Bank of Ethiopia, Ethiopian Airlines. These companies maintain high-quality English and Amharic versions of identical text.",
      color: T.accent1,
    },
    {
      title: "Local Media Subtitles & Transcripts",
      desc: "Harvesting transcripts from bilingual YouTube creators and closed-captioning files from local TV networks (Kana TV, EBS) that regularly translate international content.",
      color: T.accent2,
    },
  ];

  let cardY = 1.347;
  const cardH = 1.708;

  sections.forEach((s, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: cardY, w: 8.802, h: cardH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: cardY, w: 0.06, h: cardH, fill: { color: s.color }, line: { color: s.color } });
    slide.addText(s.title, { x: 0.905, y: cardY + 0.185, w: 8.3, h: 0.276, fontSize: 12, color: s.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(s.desc, { x: 0.905, y: cardY + 0.58, w: 8.3, h: 0.931, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
    cardY += cardH + 0.169;
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — TECH STACK (updated)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 3 \u2014 TECH STACK");
  slideTitle(slide, "Tools & Technologies");
  footerBar(slide);
  pageNum(slide, 10);

  const cols = [
    {
      abbr: "PY", color: T.accent1,
      title: "Languages & Libraries",
      items: [
        "Python 3.10+",
        "PyTorch (nn.Transformer)",
        "Pandas & NumPy",
        "SentencePiece (BPE)",
        "scikit-learn (splitting)",
        "HuggingFace datasets",
      ],
    },
    {
      abbr: "AI", color: T.accent2,
      title: "Architecture & ML",
      items: [
        "Seq2Seq Transformer",
        "AdamW optimizer",
        "Automatic Mixed Precision (AMP)",
        "SacreBLEU / chrF metrics",
        "Beam Search decoding",
        "Dynamic padding (collate_fn)",
      ],
    },
    {
      abbr: "DE", color: T.accent6,
      title: "DevOps & Tracking",
      items: [
        "FastAPI (REST API)",
        "Git & GitHub",
        "VS Code",
        "Jupyter Notebooks",
        "TensorBoard",
        "Weights & Biases",
      ],
    },
  ];

  const colX = [0.238, 3.731, 7.129];
  const colW = 3.494;
  const colH = 4.0;
  const colY = 1.348;

  cols.forEach((c, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: colX[i], y: colY, w: colW, h: colH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    pill(slide, colX[i] + 1.545, colY + 0.172, 0.401, c.color);
    slide.addText(c.abbr, { x: colX[i] + 1.58, y: colY + 0.256, w: 0.332, h: 0.224, fontSize: 11, color: T.white, bold: true, fontFace: FONT_BODY });
    slide.addText(c.title, { x: colX[i] + 0.459, y: colY + 0.695, w: 2.572, h: 0.276, fontSize: 11, color: c.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(c.items.map((item, idx) => ({
      text: "\u2022 " + item,
      options: { breakLine: idx < c.items.length - 1 }
    })), { x: colX[i] + 0.614, y: colY + 1.063, w: 2.8, h: 3.0, fontSize: 9.5, color: T.dk2, fontFace: FONT_BODY });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — DATA QUALITY
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 4 \u2014 DATA QUALITY");
  slideTitle(slide, "The \u201cGarbage In, Garbage Out\u201d Rule");
  footerBar(slide);
  pageNum(slide, 11);

  slide.addText("Raw internet data is chaotic. If we feed the network bad grammar, mixed languages, or code snippets, it will learn to output garbage.", {
    x: 0.6, y: 1.97, w: 8.641, h: 0.67, fontSize: 12, color: T.dk2, fontFace: FONT_BODY
  });

  const stages = [
    { label: "Raw Data", desc: "Messy symbols, missing rows, mixed languages, duplicates, code snippets", color: T.accent3 },
    { label: "After Cleaning", desc: "Clean, aligned Amharic\u2013English sentence pairs ready for training", color: T.accent6 },
  ];

  let boxY = 2.956;
  stages.forEach((s) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.886, y: boxY, w: 7.0, h: 0.7, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.886, y: boxY, w: 0.05, h: 0.7, fill: { color: s.color }, line: { color: s.color } });
    slide.addText(s.label, { x: 0.96, y: boxY + 0.06, w: 1.5, h: 0.276, fontSize: 11, color: s.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(s.desc, { x: 0.96, y: boxY + 0.34, w: 6.8, h: 0.282, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
    boxY += 1.038;
  });

  // Flow arrows
  slide.addText("Raw Input", { x: 8.395, y: 2.507, w: 1.116, h: 0.224, fontSize: 9, color: T.accent3, fontFace: FONT_BODY, align: "center" });
  slide.addShape(pres.shapes.RECTANGLE, { x: 8.368, y: 3.183, w: 0.865, h: 0.438, fill: { color: T.accent3 }, line: { color: T.accent3 } });
  slide.addText("Filter", { x: 8.568, y: 3.272, w: 0.668, h: 0.25, fontSize: 10, color: T.white, bold: true, fontFace: FONT_BODY, align: "center" });
  slide.addShape(pres.shapes.RECTANGLE, { x: 8.201, y: 3.957, w: 1.198, h: 0.282, fill: { color: T.accent6 }, line: { color: T.accent6 } });
  slide.addText("Clean Pairs", { x: 8.340, y: 3.983, w: 1.259, h: 0.224, fontSize: 9, color: T.white, bold: true, fontFace: FONT_BODY, align: "center" });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — CLEANING STEPS
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 4 \u2014 CLEANING STEPS");
  slideTitle(slide, "The Cleanup Operation");
  footerBar(slide);
  pageNum(slide, 12);

  const problems = [
    "UI button text (e.g., \u201cClick here / \u129b\u129a\u127b \u121a\u1275\u12ad\u122d\u1295\u201d)",
    "Missing or empty rows",
    "Duplicate paragraphs",
    "English words appearing in Amharic column",
  ];

  const solutions = [
    "Deleted empty or missing rows",
    "Length filtering: kept sentences between 3\u2013100 words",
    "Language detection to filter non-Amharic characters",
    "Deduplication & fixing spacing artifacts",
  ];

  const colW = 4.198;
  const colH = 3.995;

  // Problems
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.601, y: 1.369, w: colW, h: colH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("PROBLEMS FOUND", { x: 0.855, y: 1.498, w: 2.332, h: 0.25, fontSize: 11, color: T.accent2, bold: true, fontFace: FONT_TITLE });
  problems.forEach((p, i) => {
    const py = 2.094 + i * 0.952;
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.854, y: py, w: 0.052, h: 0.052, fill: { color: T.accent2 }, line: { color: T.accent2 } });
    slide.addText(p, { x: 1.005, y: py - 0.198, w: 4.0, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
  });

  // Solutions
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.265, y: 1.369, w: 4.611, h: colH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("SOLUTIONS APPLIED", { x: 5.455, y: 1.498, w: 2.568, h: 0.25, fontSize: 11, color: T.accent6, bold: true, fontFace: FONT_TITLE });
  solutions.forEach((s, i) => {
    const sy = 1.974 + i * 0.929;
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.454, y: sy, w: 0.052, h: 0.052, fill: { color: T.accent6 }, line: { color: T.accent6 } });
    slide.addText(s, { x: 5.605, y: sy - 0.174, w: 4.1, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — RESULTS
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 4 \u2014 RESULTS");
  slideTitle(slide, "Before vs. After");
  footerBar(slide);
  pageNum(slide, 13);

  // Before card
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: 2.002, w: 3.245, h: 1.682, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("BEFORE CLEANING", { x: 1.109, y: 2.180, w: 2.224, h: 0.224, fontSize: 10, color: T.accent2, bold: true, fontFace: FONT_TITLE });
  slide.addText("500K+", { x: 0.911, y: 2.407, w: 2.62, h: 0.839, fontSize: 48, color: T.accent2, bold: true, fontFace: FONT_TITLE });
  slide.addText("raw sentence pairs collected", { x: 0.661, y: 3.217, w: 3.121, h: 0.282, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });

  // Arrow / delta
  slide.addShape(pres.shapes.LINE, { x: 4.0, y: 2.3, w: 0, h: 1.1, line: { color: T.accent3, width: 1.5 } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 4.242, y: 2.512, w: 1.516, h: 0.662, fill: { color: T.accent4 }, line: { color: T.accent4 } });
  slide.addText("\u2212120K", { x: 4.393, y: 2.563, w: 1.214, h: 0.25, fontSize: 20, color: T.white, bold: true, fontFace: FONT_TITLE, align: "center" });
  slide.addText("unusable rows", { x: 4.393, y: 2.833, w: 1.642, h: 0.282, fontSize: 9, color: T.white, fontFace: FONT_BODY, align: "center" });

  // After card
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.156, y: 2.002, w: 3.245, h: 1.682, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("AFTER CLEANING", { x: 6.747, y: 2.180, w: 2.064, h: 0.224, fontSize: 10, color: T.accent6, bold: true, fontFace: FONT_TITLE });
  slide.addText("380K+", { x: 6.488, y: 2.407, w: 2.582, h: 0.839, fontSize: 48, color: T.accent6, bold: true, fontFace: FONT_TITLE });
  slide.addText("curated, high-quality pairs", { x: 6.324, y: 3.217, w: 2.909, h: 0.282, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });

  // Bottom note
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: 3.855, w: 8.802, h: 0.833, fill: { color: T.accent1 }, line: { color: T.accent1 } });
  slide.addText("A highly curated, dense, and effective learning environment\u2014ready for SentencePiece BPE tokenization and Transformer training.", {
    x: 0.6, y: 3.982, w: 8.8, h: 0.576, fontSize: 11, color: T.white, fontFace: FONT_BODY, align: "center"
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — PREPROCESSING SPLIT
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 5 \u2014 PREPROCESSING");
  slideTitle(slide, "Preparing for the Neural Network");
  footerBar(slide);
  pageNum(slide, 14);

  slide.addText("Converting clean text into token IDs for the Transformer. Dynamic padding per batch reduces VRAM usage during training.", {
    x: 0.6, y: 1.24, w: 8.951, h: 0.282, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });

  const splits = [
    { pct: "80%", label: "Training Data", desc: "The textbook the model studies from to learn language patterns, structure, and meaning from thousands of examples.", color: T.accent1 },
    { pct: "10%", label: "Validation Data", desc: "Practice quizzes to tune performance during training. Used for early stopping and checkpoint selection.", color: T.accent5 },
    { pct: "10%", label: "Test Data", desc: "The blind final exam to prove the model actually works on unseen data.", color: T.accent2 },
  ];

  const positions = [
    { x: 0.572, y: 1.738, w: 4.804, h: 3.342 },
    { x: 5.918, y: 1.703, w: 1.660, h: 3.338 },
    { x: 7.915, y: 1.702, w: 1.484, h: 3.339 },
  ];

  splits.forEach((s, i) => {
    const p = positions[i];
    slide.addShape(pres.shapes.RECTANGLE, { x: p.x, y: p.y, w: p.w, h: p.h, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addText(s.pct, { x: p.x, y: p.y + 0.207, w: p.w, h: 0.631, fontSize: i === 0 ? 52 : 32, color: s.color, bold: true, fontFace: FONT_TITLE, align: "center" });
    slide.addText(s.label, { x: p.x, y: p.y + 0.77, w: p.w, h: 0.276, fontSize: 11, color: s.color, bold: true, fontFace: FONT_TITLE, align: "center" });
    slide.addText(s.desc, { x: p.x + 0.1, y: p.y + 1.108, w: p.w - 0.2, h: p.h - 1.3, fontSize: 9.5, color: T.dk2, fontFace: FONT_BODY });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — TOKENIZATION (updated)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 5 \u2014 PREPROCESSING");
  slideTitle(slide, "Tokenizing Fidel (Amharic Script)");
  footerBar(slide);
  pageNum(slide, 15);

  // Ge'ez card
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.601, y: 1.346, w: 4.442, h: 2.306, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("What is Ge\u2019ez Script?", { x: 0.855, y: 1.503, w: 3.0, h: 0.276, fontSize: 12, color: T.accent1, bold: true, fontFace: FONT_TITLE });
  slide.addText("Amharic uses the ancient Ge\u2019ez alphabet, called Fidel. Unlike Latin script, each character represents a consonant-vowel combination\u2014making tokenization unique.", {
    x: 0.855, y: 1.898, w: 4.0, h: 1.582, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });

  // BPE tokenizer card
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.638, y: 3.690, w: 4.405, h: 1.845, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.638, y: 3.690, w: 0.06, h: 1.845, fill: { color: T.accent5 }, line: { color: T.accent5 } });
  slide.addText("SentencePiece BPE Tokenizer", { x: 0.818, y: 3.801, w: 3.5, h: 0.276, fontSize: 11, color: T.accent5, bold: true, fontFace: FONT_TITLE });
  slide.addText("Shared 32K vocabulary for Amharic + English. character_coverage=0.9995 handles all 200+ Ge\u2019ez symbols. <pad>=0, <unk>=1, <s>=2, </s>=3.", {
    x: 0.818, y: 4.126, w: 4.1, h: 1.3, fontSize: 9.5, color: T.dk2, fontFace: FONT_BODY
  });

  // Tokenization flow example
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.156, y: 1.352, w: 4.599, h: 4.184, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("TOKENIZATION FLOW", { x: 6.045, y: 2.093, w: 3.566, h: 0.224, fontSize: 10, color: T.accent2, bold: true, fontFace: FONT_TITLE, align: "center" });

  slide.addShape(pres.shapes.RECTANGLE, { x: 5.760, y: 2.424, w: 3.943, h: 0.917, fill: { color: T.white }, line: { color: T.accent1 } });
  slide.addText("\u1200\u121B\u12CB\u12CB\u121B", { x: 5.760, y: 2.424, w: 3.943, h: 0.60, fontSize: 32, color: T.dk2, bold: true, fontFace: "Nyala", align: "center" });
  slide.addText("Raw Amharic text", { x: 5.760, y: 3.007, w: 3.943, h: 0.282, fontSize: 10, color: T.accent3, fontFace: FONT_BODY, align: "center" });

  slide.addShape(pres.shapes.RECTANGLE, { x: 5.811, y: 3.650, w: 3.943, h: 0.807, fill: { color: T.accent1 }, line: { color: T.accent1 } });
  slide.addText("[ 3, 847, 21, 4, 12, 3 ]", { x: 5.704, y: 3.877, w: 4.0, h: 0.308, fontSize: 16, color: T.white, bold: true, fontFace: "Consolas", align: "center" });
  slide.addText("Numeric token IDs (shared vocab)", { x: 5.364, y: 4.612, w: 4.716, h: 0.606, fontSize: 9, color: T.dk2, fontFace: FONT_BODY, align: "center" });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — TOOLS / FULL STACK (updated)
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 6 \u2014 IMPLEMENTATION STATUS");
  slideTitle(slide, "My Intern Arsenal");
  footerBar(slide);
  pageNum(slide, 16);

  const tools = [
    {
      cat: "CORE",
      color: T.accent1,
      items: [
        { name: "Python", sub: "" },
        { name: "PyTorch", sub: "nn.Transformer" },
        { name: "SentencePiece", sub: "BPE tokenizer" },
      ],
    },
    {
      cat: "ML / TRACKING",
      color: T.accent2,
      items: [
        { name: "TensorBoard", sub: "Training logs" },
        { name: "Weights & Biases", sub: "Experiment tracking" },
        { name: "SacreBLEU", sub: "chrF evaluation" },
      ],
    },
    {
      cat: "DEV ENVIRONMENT",
      color: T.accent6,
      items: [
        { name: "VS Code", sub: "" },
        { name: "Jupyter Notebooks", sub: "" },
        { name: "Git & GitHub", sub: "" },
      ],
    },
  ];

  const colX = [0.601, 3.617, 6.634];
  const colW = 2.766;
  const itemH = 0.412;
  const colY = 1.348;

  tools.forEach((t, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: colX[i], y: colY, w: colW, h: 3.995, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addText(t.cat, { x: colX[i] + 0.255, y: colY + 0.150, w: 2.0, h: 0.224, fontSize: 10, color: t.color, bold: true, fontFace: FONT_TITLE });
    t.items.forEach((item, j) => {
      const iy = colY + 0.470 + j * (itemH + 0.480);
      slide.addShape(pres.shapes.RECTANGLE, { x: colX[i] + 0.255, y: iy, w: 2.255, h: itemH, fill: { color: T.white }, line: { color: T.lt2 } });
      slide.addText(item.name, { x: colX[i] + 0.405, y: iy + 0.066, w: 2.0, h: 0.276, fontSize: 11, color: T.dk2, bold: true, fontFace: FONT_TITLE });
      if (item.sub) {
        slide.addText(item.sub, { x: colX[i] + 0.405, y: iy + 0.526, w: 2.0, h: 0.276, fontSize: 9, color: t.color, fontFace: FONT_BODY });
      }
    });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — CHALLENGES
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 7 \u2014 CHALLENGES");
  slideTitle(slide, "Overcoming Obstacles");
  footerBar(slide);
  pageNum(slide, 17);

  const challenges = [
    {
      num: "1",
      color: T.accent1,
      title: "The Low-Resource Reality",
      desc: "True Amharic parallel data is incredibly sparse and often hidden in unformatted PDFs. Finding quality sources required creative thinking and systematic data collection from 4+ different sources.",
    },
    {
      num: "2",
      color: T.accent2,
      title: "Building from Scratch",
      desc: "With no prior ML experience, implementing a full Seq2Seq Transformer architecture from scratch in PyTorch meant learning positional encoding, multi-head attention, masking, AMP training, and beam search decoding all at once.",
    },
    {
      num: "3",
      color: T.accent6,
      title: "Hardware & Scale",
      desc: "Processing 380K+ rows and training a 74M parameter model requires a powerful GPU workstation. Development was split between a personal PC for pipeline prototyping and an institute workstation for full training.",
    },
  ];

  let cardY = 1.347;
  const cardH = 1.287;
  const gap = 0.141;

  challenges.forEach((c, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.599, y: cardY, w: 8.802, h: cardH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    pill(slide, 0.904, cardY + 0.162, 0.302, c.color);
    slide.addText(c.num, { x: 0.995, y: cardY + 0.253, w: 0.12, h: 0.12, fontSize: 8, color: T.white, bold: true, fontFace: FONT_BODY, align: "center" });
    slide.addText(c.title, { x: 1.405, y: cardY + 0.157, w: 7.5, h: 0.276, fontSize: 12, color: c.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(c.desc, { x: 1.405, y: cardY + 0.512, w: 7.9, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY });
    cardY += cardH + gap;
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18 — TAKEAWAYS
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 8 \u2014 TAKEAWAYS");
  slideTitle(slide, "Lessons from the Trenches");
  footerBar(slide);
  pageNum(slide, 18);

  // Left card
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.635, y: 1.348, w: 4.198, h: 3.995, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("THE ML REALITY CHECK", { x: 0.905, y: 1.554, w: 3.0, h: 0.224, fontSize: 10, color: T.accent1, bold: true, fontFace: FONT_TITLE });
  slide.addText("80%", { x: 0.905, y: 1.761, w: 1.669, h: 0.979, fontSize: 52, color: T.accent1, bold: true, fontFace: FONT_TITLE });
  slide.addText("of ML work is Data Engineering", { x: 2.574, y: 1.922, w: 2.437, h: 0.576, fontSize: 11, color: T.dk2, fontFace: FONT_TITLE });
  slide.addText("Finding, cleaning, and managing data is the bulk of the job. The model training itself? That\u2019s the easy part. This is the industry\u2019s best-kept secret.", {
    x: 0.905, y: 3.010, w: 4.352, h: 1.257, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });

  // Right card
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.201, y: 1.348, w: 4.198, h: 3.995, fill: { color: T.lt2 }, line: { color: T.lt2 } });
  slide.addText("PERSONAL GROWTH", { x: 5.505, y: 1.554, w: 3.0, h: 0.224, fontSize: 10, color: T.accent2, bold: true, fontFace: FONT_TITLE });
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.506, y: 2.129, w: 0.052, h: 0.052, fill: { color: T.accent2 }, line: { color: T.accent2 } });
  slide.addText("Transitioned from IT theory to practical, real-world AI pipelines", {
    x: 5.705, y: 1.922, w: 4.363, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.506, y: 2.898, w: 0.052, h: 0.052, fill: { color: T.accent5 }, line: { color: T.accent5 } });
  slide.addText("Learned to work with messy, unstructured data at scale", {
    x: 5.705, y: 2.685, w: 3.5, h: 0.606, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.506, y: 3.633, w: 0.052, h: 0.052, fill: { color: T.accent6 }, line: { color: T.accent6 } });
  slide.addText("Built professional DevOps habits\u2014version control, environment management, GPU training workflows", {
    x: 5.705, y: 3.447, w: 3.839, h: 0.932, fontSize: 10, color: T.dk2, fontFace: FONT_BODY
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 19 — PROJECT ROADMAP (replaces "next steps")
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);
  sectionLabel(slide, "SECTION 8 \u2014 ROADMAP");
  slideTitle(slide, "Project Roadmap: Implementation Phases");
  footerBar(slide);
  pageNum(slide, 19);

  const phases = [
    { num: "1", label: "Data Collection", status: "DONE", color: T.accent6, desc: "380K+ curated sentence pairs" },
    { num: "2", label: "Preprocessing & BPE", status: "DONE", color: T.accent6, desc: "Normalisation, 95/2.5/2.5 split, SentencePiece 32K vocab" },
    { num: "3", label: "Transformer Architecture", status: "DONE", color: T.accent6, desc: "Seq2Seq Transformer (74M params), plumbing test passed" },
    { num: "4", label: "Training Pipeline", status: "UPCOMING", color: T.accent1, desc: "AMP, AdamW, warmup, early stopping, checkpointing" },
    { num: "5", label: "Beam Search & Evaluation", status: "UPCOMING", color: T.accent2, desc: "Greedy + Beam Search (width=5), BLEU & chrF scores" },
    { num: "6", label: "FastAPI Deployment", status: "UPCOMING", color: T.accent5, desc: "/health, /translate endpoints, torch.no_grad()" },
  ];

  const cardW = 4.198;
  const cardH = 1.286;
  const colGap = 0.141;
  const positions = [
    { x: 0.599, y: 1.346 }, { x: 5.201, y: 1.346 },
    { x: 0.599, y: 2.773 }, { x: 5.201, y: 2.773 },
    { x: 0.599, y: 4.200 }, { x: 5.201, y: 4.200 },
  ];

  phases.forEach((p, i) => {
    const pos = positions[i];
    slide.addShape(pres.shapes.RECTANGLE, { x: pos.x, y: pos.y, w: cardW, h: cardH, fill: { color: T.lt2 }, line: { color: T.lt2 } });
    slide.addShape(pres.shapes.RECTANGLE, { x: pos.x, y: pos.y, w: 0.06, h: cardH, fill: { color: p.color }, line: { color: p.color } });
    pill(slide, pos.x + 0.305, pos.y + 0.162, 0.302, p.color);
    slide.addText(p.num, { x: pos.x + 0.415, y: pos.y + 0.206, w: 0.2, h: 0.208, fontSize: 11, color: T.white, bold: true, fontFace: FONT_BODY, align: "center" });
    slide.addText(p.label, { x: pos.x + 0.755, y: pos.y + 0.157, w: 2.5, h: 0.276, fontSize: 11, color: p.color, bold: true, fontFace: FONT_TITLE });
    slide.addText(p.desc, { x: pos.x + 0.755, y: pos.y + 0.457, w: 3.3, h: 0.606, fontSize: 9, color: T.dk2, fontFace: FONT_BODY });
    slide.addText(p.status, { x: pos.x + cardW - 1.0, y: pos.y + cardH - 0.36, w: 0.9, h: 0.25, fontSize: 8, color: p.color, bold: true, fontFace: FONT_BODY, align: "right" });
  });
})();


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 20 — THANK YOU
// ════════════════════════════════════════════════════════════════════════════
(function() {
  const slide = pres.addSlide();
  topBar(slide);

  slide.addText("Amharic\u2013English NMT \u2014 Phases 1\u20133 Complete", {
    x: 1.770, y: 1.020, w: 6.46, h: 0.224, fontSize: 12, color: T.accent1, bold: true, fontFace: FONT_TITLE, align: "center"
  });

  slide.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 1.368, w: 0.5, h: 0.016, fill: { color: T.accent1 }, line: { color: T.accent1 } });

  slide.addText("Thank You", {
    x: 2.939, y: 1.506, w: 4.122, h: 0.839, fontSize: 48, color: T.dk2, bold: true, fontFace: FONT_TITLE, align: "center"
  });

  slide.addText("I\u2019d love to take any questions or hear your feedback.", {
    x: 1.599, y: 2.426, w: 6.802, h: 0.339, fontSize: 14, color: T.dk2, fontFace: FONT_BODY, align: "center"
  });

  slide.addShape(pres.shapes.RECTANGLE, { x: 4.749, y: 3.225, w: 0.5, h: 0.016, fill: { color: T.accent1 }, line: { color: T.accent1 } });

  slide.addText("Yohannes Zewde \u00b7 IT Student & ML Intern", {
    x: 2.958, y: 3.515, w: 4.085, h: 0.282, fontSize: 11, color: T.dk2, fontFace: FONT_BODY, align: "center"
  });
  footerBar(slide);
})();


// ════════════════════════════════════════════════════════════════════════════
// WRITE FILE
// ════════════════════════════════════════════════════════════════════════════
pres.writeFile({ fileName: "./output/amharic-nmt-phases1-3.pptx" })
  .then(() => console.log("Done: slides/output/amharic-nmt-phases1-3.pptx"))
  .catch(err => { console.error(err); process.exit(1); });
