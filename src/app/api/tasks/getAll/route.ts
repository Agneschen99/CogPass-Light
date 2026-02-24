import { NextResponse } from "next/server";
import { getTasks } from "../data";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ tasks: getTasks() });
}
