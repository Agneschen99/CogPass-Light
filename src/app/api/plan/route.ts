// src/app/api/plan/route.ts
import { NextRequest } from "next/server";

export const dynamic = "force-dynamic"; // 避免被缓存
export const revalidate = 0;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json(); // { tasks: [...], options?: {...} }

    const base = process.env.MODEL_BACKEND_URL?.replace(/\/+$/, "");
    const token = process.env.MODEL_TOKEN ?? "";

    // 如果暂时还没有后端地址，直接返回一个 mock，让前端能跑通
    if (!base) {
      return Response.json(mockPlan(body), { status: 200 });
    }

    const resp = await fetch(`${base}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      // 明确不要缓存
      cache: "no-store",
      // 如果你后端响应较慢可加超时控制（可选）
      // @ts-ignore
      next: { revalidate: 0 },
    });

    if (!resp.ok) {
      const text = await resp.text();
      return new Response(text || "Backend error", { status: resp.status || 500 });
    }

    // 后端请直接返回 JSON 文本（字符串），这里原样转发
    const text = await resp.text();
    return new Response(text, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    console.error("[/api/plan] error:", err);
    return Response.json({ error: err?.message ?? "Server error" }, { status: 500 });
  }
}

/** 没有后端时的兜底：返回一份可渲染的周计划结构 */
function mockPlan(_body: any) {
  const now = new Date();
  const start = new Date(now);
  start.setHours(9, 0, 0, 0);

  return {
    todayTop3: [
      {
        id: "demo-top1",
        title: "Read Chapter 1",
        subject: "reading",
        start: start.toISOString(),
        minutes: 45,
        done: false,
      },
      {
        id: "demo-top2",
        title: "Math practice",
        subject: "math",
        start: new Date(start.getTime() + 60 * 60 * 1000).toISOString(),
        minutes: 30,
        done: false,
      },
      {
        id: "demo-top3",
        title: "Review notes",
        subject: "general",
        start: new Date(start.getTime() + 120 * 60 * 1000).toISOString(),
        minutes: 25,
        done: false,
      },
    ],
    weekSlots: [
      {
        id: "demo-1",
        title: "Deep work",
        subject: "focus",
        start: start.toISOString(),
        minutes: 60,
        done: false,
      },
      {
        id: "demo-2",
        title: "Lecture",
        subject: "physics",
        start: new Date(start.getTime() + 3 * 60 * 60 * 1000).toISOString(),
        minutes: 90,
        done: false,
      },
    ],
  };
}
