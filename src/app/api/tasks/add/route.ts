import { NextResponse } from "next/server";
import { addTask, getTasks } from "../data";
import type { Task } from "@/app/light/types";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    const payload = (await req.json()) as Partial<Task>;
    if (!payload?.title || typeof payload.title !== "string") {
      return NextResponse.json({ error: "title is required" }, { status: 400 });
    }
    const estimatedTime = Number(payload.estimatedTime ?? 0);
    if (!Number.isFinite(estimatedTime) || estimatedTime <= 0) {
      return NextResponse.json({ error: "estimatedTime must be > 0" }, { status: 400 });
    }

    const task = addTask({
      title: payload.title.trim(),
      dueDate: payload.dueDate,
      estimatedTime,
    });

    return NextResponse.json({ task, tasks: getTasks() });
  } catch (err) {
    return NextResponse.json({ error: "invalid payload" }, { status: 400 });
  }
}
