import { NextResponse } from "next/server";
import { getTop3, setTop3, removeFromTop3 } from "../data";

export const dynamic = "force-dynamic";

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function daysUntil(dateStr?: string) {
  if (!dateStr) return null;
  const due = new Date(dateStr);
  if (Number.isNaN(due.getTime())) return null;

  const now = new Date();
  // 只算日期，不算时分秒（更稳定）
  const d0 = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const d1 = new Date(due.getFullYear(), due.getMonth(), due.getDate());
  const diffMs = d1.getTime() - d0.getTime();
  return Math.round(diffMs / (1000 * 60 * 60 * 24));
}

function loadScore(category?: string) {
  // deep > normal > quick
  if (category === "deep") return 3;
  if (category === "quick") return 1;
  return 2; // normal / undefined 默认 normal
}

function urgencyScore(dueDate?: string) {
  const d = daysUntil(dueDate);
  if (d === null) return 0.2;          // 没截止日期也给一点点分
  if (d < 0) return 3.0;               // 过期：最高
  if (d === 0) return 2.6;             // 今天
  if (d <= 1) return 2.2;              // 明天
  if (d <= 3) return 1.6;
  if (d <= 7) return 1.0;
  return 0.5;
}

function timeScore(estimatedTime?: number) {
  // 让“合理时长”加一点分：25-60 分钟更像 Deep work slot
  const t = estimatedTime ?? 25;
  if (t >= 25 && t <= 60) return 0.6;
  if (t < 15) return 0.2;
  if (t > 90) return 0.1;
  return 0.4;
}

function pickTop3(tasks: any[]) {
  const scored = tasks.map((t) => {
    const l = loadScore(t.category);
    const u = urgencyScore(t.dueDate);
    const ts = timeScore(t.estimatedTime);

    // 你之前设想的权重：Load 0.4 + Urgency 0.3 + Energy 0.3
    // 这里先不做 Energy（等你接 chronotype 再加）
    const score = l * 0.55 + u * 0.35 + ts * 0.10;

    return { task: t, score };
  });

  scored.sort((a, b) => b.score - a.score);

  // 保证只要 3 个
  return scored.slice(0, 3).map((x) => x.task);
}

export async function GET() {
  return NextResponse.json({ tasks: getTop3() });
}

export async function POST(req: Request) {
  try {
    const { id } = (await req.json()) as { id?: string };
    if (!id) return NextResponse.json({ error: "id is required" }, { status: 400 });
    const top3 = setTop3(id);
    return NextResponse.json(top3);
  } catch {
    return NextResponse.json({ error: "invalid payload" }, { status: 400 });
  }
}

// ✅ 新增：从 Top3 移除
export async function DELETE(req: Request) {
  try {
    const { id } = (await req.json()) as { id?: string };
    if (!id) return NextResponse.json({ error: "id is required" }, { status: 400 });
    const top3 = removeFromTop3(id);
    return NextResponse.json(top3);
  } catch {
    return NextResponse.json({ error: "invalid payload" }, { status: 400 });
  }
}
