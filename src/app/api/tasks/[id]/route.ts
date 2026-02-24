// src/app/api/tasks/[id]/route.ts
import { NextResponse } from "next/server";
import { deleteTask } from "../data";

export const dynamic = "force-dynamic";

export async function DELETE(
  _req: Request,
  { params }: { params: { id: string } }
) {
  const { id } = params;
  const result = deleteTask(id);
  return NextResponse.json(result);
}