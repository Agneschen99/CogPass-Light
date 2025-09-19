// src/app/api/plan/route.ts
import { NextRequest, NextResponse } from "next/server";
import { generatePlan } from "@/lib/planner";

export const dynamic = "force-dynamic"; 
export const revalidate = 0;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json(); // { tasks: [...], options?: {...} }
    const plan = generatePlan(body.tasks ?? [], body.options?.dailyMaxBlocks);
    return NextResponse.json(plan, { status: 200 });
  } catch (err: any) {
    console.error("[/api/plan] error:", err);
    return NextResponse.json(
      { error: err?.message ?? "Server error" },
      { status: 500 }
    );
  }
}
