import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

process.env.HOME = process.env.HOME || "C:/Users/User";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SLIDE = { width: 1280, height: 720 };
const C = {
  bg: "#F5F7FB",
  white: "#FFFFFF",
  ink: "#112033",
  sub: "#495A70",
  blue: "#205CFF",
  line: "#D8E0EA",
  blueSoft: "#EAF1FF",
  greenSoft: "#EAF8EF",
  amberSoft: "#FFF4DF",
  graySoft: "#EEF2F7",
};

function addBox(slide, left, top, width, height, fill = C.white, line = C.line, radius = 10) {
  const shape = slide.shapes.add({
    geometry: radius > 0 ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill: { type: "solid", color: fill },
    line: { style: "solid", fill: line, width: line === fill ? 0 : 1 },
  });
  if (radius > 0) shape.borderRadius = radius;
  return shape;
}

function addText(slide, left, top, width, height, value, opts = {}) {
  const {
    fontSize = 18,
    color = C.ink,
    bold = false,
    fill = C.bg,
    line = C.bg,
    align = "left",
    valign = "top",
    inset = 0,
  } = opts;
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: { type: "solid", color: fill },
    line: { style: "solid", fill: line, width: line === fill ? 0 : 1 },
  });
  shape.text.style = {
    typeface: "Microsoft YaHei",
    fontSize,
    color,
    bold,
    alignment: align,
    verticalAlignment: valign,
    insets: { left: inset, right: inset, top: inset, bottom: inset },
  };
  shape.text = value;
  return shape;
}

function addTitle(slide, titleText, subText = "") {
  addText(slide, 72, 40, 1120, 22, "汇报提纲", { fontSize: 15, color: C.blue, bold: true });
  addText(slide, 72, 68, 1136, 46, titleText, { fontSize: 30, bold: true });
  if (subText) addText(slide, 72, 120, 1136, 34, subText, { fontSize: 15, color: C.sub });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 72, top: 162, width: 1136, height: 2 },
    fill: { type: "solid", color: C.line },
    line: { style: "solid", fill: C.line, width: 0 },
  });
}

function addFooter(slide, value) {
  addText(slide, 72, 690, 1120, 14, value, { fontSize: 11, color: "#8092A7" });
}

function bullets(items) {
  return items.map((x) => `• ${x}`).join("\n");
}

function addSectionCard(slide, x, y, w, h, head, body, fill = C.white) {
  addBox(slide, x, y, w, h, fill);
  addText(slide, x + 18, y + 16, w - 36, 26, head, {
    fontSize: 20,
    bold: true,
    fill,
    line: fill,
  });
  addText(slide, x + 18, y + 56, w - 36, h - 72, body, {
    fontSize: 17,
    color: C.sub,
    fill,
    line: fill,
  });
}

function addChapterSlide(slide, no, heading, detail) {
  addTitle(slide, `第${no}章 ${heading}`);
  addBox(slide, 92, 220, 1096, 282, C.blueSoft, C.blueSoft, 18);
  addText(slide, 120, 268, 110, 56, `${no}`, {
    fontSize: 52,
    bold: true,
    fill: C.blueSoft,
    line: C.blueSoft,
  });
  addText(slide, 236, 270, 830, 48, heading, {
    fontSize: 34,
    bold: true,
    fill: C.blueSoft,
    line: C.blueSoft,
  });
  addText(slide, 122, 360, 990, 96, detail, {
    fontSize: 22,
    color: C.sub,
    fill: C.blueSoft,
    line: C.blueSoft,
  });
}

function addCompareRow(slide, y, label, a, b) {
  addText(slide, 82, y, 290, 40, label, { fontSize: 17, bold: true });
  const fillA = a === "强" ? C.greenSoft : a === "中" ? C.amberSoft : C.graySoft;
  const fillB = b === "强" ? C.greenSoft : b === "中" ? C.amberSoft : C.graySoft;
  addBox(slide, 468, y - 1, 170, 40, fillA, fillA, 14);
  addBox(slide, 856, y - 1, 170, 40, fillB, fillB, 14);
  addText(slide, 468, y + 7, 170, 20, a, { fontSize: 16, bold: true, align: "center", fill: fillA, line: fillA });
  addText(slide, 856, y + 7, 170, 20, b, { fontSize: 16, bold: true, align: "center", fill: fillB, line: fillB });
}

function slide01(p) {
  const s = p.slides.add();
  addTitle(s, "论文组织与方法提取汇报", "围绕原论文、已完成工作和方法提取方案进行正式梳理");
  addSectionCard(s, 72, 210, 352, 252, "第一章 论文总结", bullets([
    "原论文的研究背景",
    "原论文的方法主线",
    "原论文的贡献点与结构特点",
  ]), C.blueSoft);
  addSectionCard(s, 464, 210, 352, 252, "第二章 已完成工作", bullets([
    "统一问题层与联邦主线",
    "神经侧改进模块",
    "当前已经稳定成立的结论",
  ]), C.greenSoft);
  addSectionCard(s, 856, 210, 352, 252, "第三章 提取方法", bullets([
    "方案一：只保留神经网络主线",
    "方案二：保留线性模型，分成两种模型实现",
    "比较两种方案的组织方式",
  ]), C.amberSoft);
  addBox(s, 72, 506, 1136, 116, C.white);
  addText(s, 96, 530, 1080, 26, "汇报目标", { fontSize: 20, bold: true, fill: C.white, line: C.white });
  addText(s, 96, 568, 1080, 36, "对现有工作进行结构化整理，明确论文组织方式与方法主语，形成可直接进入写作的方案。", {
    fontSize: 20,
    color: C.sub,
    fill: C.white,
    line: C.white,
  });
  addFooter(s, "整体结构：论文总结 → 已完成工作 → 提取方法");
  return s;
}

function slide02(p) {
  const s = p.slides.add();
  addChapterSlide(s, "一", "论文总结", "本章只总结原论文本身：研究背景是什么，方法主线是什么，贡献点是什么，以及其结构为什么显得清楚。");
  addFooter(s, "第一章");
  return s;
}

function slide03(p) {
  const s = p.slides.add();
  addTitle(s, "原论文背景：为什么会提出这个问题");
  addSectionCard(s, 72, 196, 350, 388, "现实背景", bullets([
    "负荷预测模型已经训练完成并投入使用。",
    "训练数据中可能出现隐私撤回、错误样本或恶意样本。",
    "需要删除这些样本对模型参数和预测行为造成的影响。",
  ]), C.blueSoft);
  addSectionCard(s, 466, 196, 350, 388, "直接重训的问题", bullets([
    "重训成本高，效率低。",
    "原始训练数据未必始终完整可用。",
    "工程上更希望在现有模型基础上直接修复。",
  ]), C.graySoft);
  addSectionCard(s, 860, 196, 348, 388, "问题转化", bullets([
    "机器遗忘的目标是让模型忘掉指定样本。",
    "直接遗忘会导致模型性能下降。",
    "因此必须在遗忘后继续进行修复。",
  ]), C.greenSoft);
  addFooter(s, "第一章：原论文背景");
  return s;
}

function slide04(p) {
  const s = p.slides.add();
  addTitle(s, "原论文方法：同一条修复流程，使用不同修复目标");
  addBox(s, 72, 196, 1136, 132, C.white);
  addText(s, 94, 222, 1088, 82, "遗忘请求 → 完整遗忘 → 计算测试目标梯度 → 计算参数敏感方向 → 对保留样本打分 → 约束重加权修复", {
    fontSize: 24,
    fill: C.white,
    line: C.white,
  });
  addSectionCard(s, 72, 360, 538, 250, "性能导向修复", bullets([
    "修复目标是统计性能。",
    "典型指标包括均方误差和平均绝对百分比误差。",
    "核心问题是：哪些保留样本更有利于恢复统计指标。",
  ]), C.blueSoft);
  addSectionCard(s, 670, 360, 538, 250, "任务导向修复", bullets([
    "修复目标是下游运行成本。",
    "作者强调统计误差不能完整代表实际运行效果。",
    "核心问题是：哪些保留样本更有利于恢复运行成本。",
  ]), C.greenSoft);
  addFooter(s, "第一章：原论文方法主线");
  return s;
}

function slide05(p) {
  const s = p.slides.add();
  addTitle(s, "原论文的贡献点与结构特点");
  addSectionCard(s, 72, 196, 540, 402, "原论文的主要贡献", bullets([
    "将机器遗忘引入负荷预测场景。",
    "提出性能导向修复：通过影响函数和样本重加权保持统计性能。",
    "提出任务导向修复：直接以下游运行成本作为修复目标。",
    "证明任务导向目标可以进入重加权修复流程。",
  ]), C.blueSoft);
  addSectionCard(s, 668, 196, 540, 402, "原论文为什么显得清楚", bullets([
    "主语始终是修复目标，而不是模型结构。",
    "先用线性模型给出最清楚的解释对象，再扩展到卷积模型和混合模型。",
    "公式、实验和图表都围绕同一个中心问题展开。",
    "模型是验证对象，不是方法主语。",
  ]), C.amberSoft);
  addFooter(s, "第一章：原论文贡献与结构");
  return s;
}

function slide06(p) {
  const s = p.slides.add();
  addChapterSlide(s, "二", "已完成工作", "本章按照论文写作方式，对现有工作进行系统梳理，只讨论已经完成并反复验证过的内容。");
  addFooter(s, "第二章");
  return s;
}

function slide07(p) {
  const s = p.slides.add();
  addTitle(s, "统一问题层与联邦主线");
  addSectionCard(s, 72, 196, 350, 332, "统一问题层", bullets([
    "已经固定统一的遗忘设置：待遗忘样本、保留样本、测试样本。",
    "已经固定统一的修复问题：完整遗忘之后继续进行修复。",
    "已经固定统一的修复准则集合：均方误差、平均绝对百分比误差、运行成本。",
    "已经形成固定对象下只改变修复准则的比较方式。",
  ]), C.blueSoft);
  addSectionCard(s, 466, 196, 350, 332, "联邦主线", bullets([
    "云端负责修复准则对应的测试目标信号。",
    "客户端负责局部数据、局部贡献和局部可计算对象。",
    "联邦主线已经打通：测试目标 → 敏感方向 → 样本得分 → 重加权修复。",
    "因此已经具备一条完整的联邦修复流程，而不是零散实验。",
  ]), C.greenSoft);
  addSectionCard(s, 860, 196, 348, 332, "这一部分的重要性", bullets([
    "它决定论文的方法主干是否存在。",
    "它说明现有工作已经不只是实验堆叠。",
    "后续所有模型、模块和结果都建立在这条主线上。",
    "方法提取时，真正需要保留的骨架就是这一层。",
  ]), C.graySoft);
  addBox(s, 72, 548, 1136, 106, C.white);
  addText(s, 92, 568, 1080, 22, "统一主线的公式骨架", { fontSize: 18, bold: true, fill: C.white, line: C.white });
  addText(s, 92, 600, 1080, 34, "测试目标梯度  g_te = ∇θ L_rep  →  参数敏感方向  M = - H^-1 g_te  →  样本得分  s_i = ∇θ l_i^T M  →  受约束重加权修复", {
    fontSize: 18,
    color: C.sub,
    fill: C.white,
    line: C.white,
  });
  addFooter(s, "第二章：统一问题层与联邦主线");
  return s;
}

function slide08(p) {
  const s = p.slides.add();
  addTitle(s, "联邦主线与原论文方法的差异");
  addSectionCard(s, 72, 196, 350, 324, "原论文的方法主线", bullets([
    "原论文的修复链条建立在集中式对象上。",
    "测试目标梯度、参数敏感方向和样本得分都直接围绕单机模型展开。",
    "线性模型先作为最清楚的解释对象，再扩展到卷积模型和混合模型。",
    "主语始终是修复目标本身，而不是实现结构。",
  ]), C.blueSoft);
  addSectionCard(s, 466, 196, 350, 324, "我们已完成的联邦主线", bullets([
    "修复链条不再建立在单机对象上，而是建立在云端与客户端分工上。",
    "云端负责测试目标信号、全局聚合和修复求解，客户端负责局部数据、局部贡献和局部可计算对象。",
    "敏感方向与样本得分不再直接按原论文公式整机求解，而是改写成联邦可计算形式。",
    "主语已经从修复目标进一步转向联邦可实现的修复流程。",
  ]), C.greenSoft);
  addSectionCard(s, 860, 196, 348, 324, "这一差异的实际含义", bullets([
    "相同的是：修复问题、修复准则和重加权思路仍然保留。",
    "不同的是：可计算对象、信息流向和实现骨架已经被改写。",
    "因此我们现在完成的内容，已经不是简单复现，而是一条新的联邦修复实现路线。",
    "后续若提取为方法，真正应当突出的就是这条联邦实现主线。",
  ]), C.amberSoft);
  addBox(s, 72, 520, 538, 136, C.white);
  addText(s, 92, 540, 490, 22, "原论文的核心形式", { fontSize: 18, bold: true, fill: C.white, line: C.white });
  addText(s, 92, 574, 490, 58, "g_te = ∇θ L_rep\nM = - H^-1 g_te\ns_i = ∇θ l_i^T M", {
    fontSize: 20,
    color: C.sub,
    fill: C.white,
    line: C.white,
  });
  addBox(s, 670, 520, 538, 136, C.white);
  addText(s, 690, 540, 490, 22, "联邦改写后的核心形式", { fontSize: 18, bold: true, fill: C.white, line: C.white });
  addText(s, 690, 574, 490, 58, "g_te 由云端计算\nM 由云端目标信号与客户端局部对象联合形成\ns_i = Σ_k s_i^(k)，由客户端局部贡献聚合得到", {
    fontSize: 18,
    color: C.sub,
    fill: C.white,
    line: C.white,
  });
  addFooter(s, "第二章：联邦主线与原论文的差异");
  return s;
}

function slide09(p) {
  const s = p.slides.add();
  addTitle(s, "神经侧改进模块");
  addSectionCard(s, 72, 196, 350, 388, "拓扑模块", bullets([
    "拓扑感知划分。",
    "拓扑正则先验。",
    "作用是改善客户端局部视图与修复对象的结构先验。",
  ]), C.blueSoft);
  addSectionCard(s, 466, 196, 350, 388, "服务器侧聚合特征接口", bullets([
    "最终保留的形式是聚合均值接口。",
    "作用是改变服务器侧修复对象的输入形式。",
    "在卷积模型上对成本导向修复有明显改善。",
  ]), C.greenSoft);
  addSectionCard(s, 860, 196, 348, 388, "成本导向影子权重", bullets([
    "适合作为成本增强模块。",
    "对成本指标稳定有效。",
    "对统计指标存在明显边界。",
  ]), C.amberSoft);
  addFooter(s, "第二章：神经侧改进模块");
  return s;
}

function slide10(p) {
  const s = p.slides.add();
  addTitle(s, "当前已经稳定成立的结论");
  addSectionCard(s, 72, 196, 540, 404, "已经可以进入正文的稳定结论", bullets([
    "联邦修复主线已经完整成立。",
    "卷积模型与混合模型都形成了可运行、可比较、可增强的分支。",
    "拓扑模块与聚合接口属于基础改善层。",
    "影子权重最适合写成成本导向增强模块。",
  ]), C.greenSoft);
  addSectionCard(s, 668, 196, 540, 404, "当前组织上的主要问题", bullets([
    "原论文的主语是修复目标，而现有工作主语更像联邦实现。",
    "线性模型和神经模型的实现路线并不完全对称。",
    "如果不重新组织，拓扑、接口、影子权重会显得像外加模块。",
    "因此现阶段重点应当是提取方法，而不是继续堆叠实验叙事。",
  ]), C.amberSoft);
  addFooter(s, "第二章：稳定结论");
  return s;
}

function slide11(p) {
  const s = p.slides.add();
  addChapterSlide(s, "三", "提取方法", "本章只讨论论文组织方案，不讨论新的实验。当前只考虑两种方案：只保留神经网络主线；或保留线性模型并分成两种模型实现。");
  addFooter(s, "第三章");
  return s;
}

function slide12(p) {
  const s = p.slides.add();
  addTitle(s, "方案一：只保留神经网络主线");
  addSectionCard(s, 72, 196, 540, 404, "组织方式", bullets([
    "直接提出一个新的联邦方法，例如联邦准则感知机器遗忘方法。",
    "方法主语固定为云边解耦的联邦修复框架。",
    "正文只保留卷积模型与混合模型两类神经骨干。",
    "拓扑模块、聚合接口、影子权重统一写成神经侧改进层。",
  ]), C.blueSoft);
  addSectionCard(s, 668, 196, 540, 404, "优点与风险", bullets([
    "优点：方法主语最明确，最像独立的新方法论文。",
    "优点：不需要解释线性模型和神经模型的不对称结构。",
    "风险：失去原论文从线性模型引入神经网络的自然过渡。",
    "风险：需要明确限定方法主要面向联邦神经负荷预测器。",
  ]), C.greenSoft);
  addFooter(s, "第三章：方案一");
  return s;
}

function slide13(p) {
  const s = p.slides.add();
  addTitle(s, "方案二：保留线性模型，分成两种模型实现");
  addSectionCard(s, 72, 196, 540, 404, "组织方式", bullets([
    "总方法主语仍然放在新的联邦方法上。",
    "再将实现层划分为线性统计量实现和神经分块实现。",
    "正文强调：统一的是方法框架，不统一的是运行时实现。",
    "线性模型作为解释性与解析基线存在，但不抢占方法主语。",
  ]), C.blueSoft);
  addSectionCard(s, 668, 196, 540, 404, "优点与风险", bullets([
    "优点：与原论文之间保留更自然的桥梁。",
    "优点：可以更清楚交代线性模型的存在价值。",
    "风险：论文结构会比方案一更复杂。",
    "风险：如果组织不当，线性模型和神经模型仍会显得像两条路线。",
  ]), C.amberSoft);
  addFooter(s, "第三章：方案二");
  return s;
}

function slide14(p) {
  const s = p.slides.add();
  addTitle(s, "两种方案比较");
  addText(s, 492, 188, 120, 24, "方案一", { fontSize: 16, color: C.sub, bold: true, align: "center" });
  addText(s, 880, 188, 120, 24, "方案二", { fontSize: 16, color: C.sub, bold: true, align: "center" });
  addCompareRow(s, 242, "方法主语明确程度", "强", "中");
  addCompareRow(s, 310, "主线统一程度", "强", "中");
  addCompareRow(s, 378, "与原论文连续性", "中", "强");
  addCompareRow(s, 446, "结构复杂度控制", "强", "中");
  addCompareRow(s, 514, "当前推荐程度", "强", "中");
  addBox(s, 72, 584, 1136, 72, C.blueSoft, C.blueSoft, 18);
  addText(s, 94, 606, 1080, 26, "建议优先采用方案一：只保留神经网络主线。若需要保留与原论文更强的连续性，再考虑方案二。", {
    fontSize: 19,
    bold: true,
    fill: C.blueSoft,
    line: C.blueSoft,
  });
  addFooter(s, "第三章：两种方案比较");
  return s;
}

async function saveBlob(blob, outPath) {
  const ab = await blob.arrayBuffer();
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, Buffer.from(ab));
}

async function main() {
  const outDir = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve("deliverables");
  const outPptx = path.join(outDir, "fedcau_official_report_v5.pptx");
  const previewDir = path.join(outDir, "fedcau_official_report_v5_preview");
  const p = Presentation.create({ slideSize: SLIDE });
  const slides = [
    slide01(p),
    slide02(p),
    slide03(p),
    slide04(p),
    slide05(p),
    slide06(p),
    slide07(p),
    slide08(p),
    slide09(p),
    slide10(p),
    slide11(p),
    slide12(p),
    slide13(p),
    slide14(p),
  ];

  await fs.mkdir(previewDir, { recursive: true });
  for (let i = 0; i < slides.length; i += 1) {
    const png = await p.export({ slide: slides[i], format: "png", scale: 1 });
    await saveBlob(png, path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }

  const pptx = await PresentationFile.exportPptx(p);
  await fs.mkdir(outDir, { recursive: true });
  await pptx.save(outPptx);
  console.log(JSON.stringify({ outPptx, previewDir, slideCount: slides.length }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
