// src/app/light-mode/page.tsx
"use client";

import React, { useMemo, useState } from "react";
import dayjs from "dayjs";
import dynamic from "next/dynamic";
import TaskList from "@/components/TaskList";
import Top3List from "@/components/Top3List";
import { useStore } from "@/state/store";

const Calendar = dynamic(() => import("@/components/Calendar"), { ssr: false });

export default function LightModePage() {
  const { tasks, todayTop3, addTask } = useStore();
  const [quickInput, setQuickInput] = useState("");

  function handleQuickAdd() {
    const title = quickInput.trim();
    if (!title) return;
    addTask({
      title,
      subject: "general",
      difficulty: 3,
      minutes: 25,
      due: dayjs().format("YYYY-MM-DD"),
    });
    setQuickInput("");
  }

  const calendarEvents = useMemo(() => {
    const taskEvents =
      tasks?.filter((t: any) => t.due).map((t: any, i: number) => {
        const start = dayjs(t.due || dayjs()).hour(9).minute(0);
        const end = start.add(t.minutes ?? 25, "minute");
        return {
          id: t.id ?? `task-${i}`,
          title: t.title || "Task",
          start: start.toISOString(),
          end: end.toISOString(),
          extendedProps: { kind: "task" },
        };
      }) ?? [];

    const top3Events =
      todayTop3?.map((t: any, i: number) => {
        const startTime = t.start
          ? dayjs(t.start)
          : dayjs().hour(10 + i).minute(0);
        const endTime = t.end
          ? dayjs(t.end)
          : startTime.add(t.minutes ?? 25, "minute");

        return {
          id: t.id ?? `top3-${i}`,
          title: t.title || "Top Task",
          start: startTime.toISOString(),
          end: endTime.toISOString(),
          extendedProps: { kind: "slot" },
        };
      }) ?? [];

    return [...top3Events, ...taskEvents];
  }, [tasks, todayTop3]);

  return (
    <div className="min-h-screen bg-[#FAFAFA] p-6">
      {/* 标题区 */}
      <header className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-800">☀️ NeuroPlan · Light Mode</h1>
        <div className="text-sm text-gray-500">AI Planning System</div>
      </header>

      {/* 核心布局：左边任务，右边日历 */}
      <div className="flex flex-col gap-6 lg:flex-row">
        {/* 左侧：任务管理 (队友的部分 + 你的Chat) */}
        <div className="flex w-full flex-col gap-6 lg:w-1/3">
          {/* 这里可以放你的 Chat Input */}
          <div className="rounded-xl border bg-white p-4 shadow-sm">
            <h3 className="mb-2 font-semibold">Quick Add</h3>
            <div className="flex gap-2">
              <input
                type="text"
                value={quickInput}
                onChange={(e) => setQuickInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleQuickAdd();
                }}
                placeholder="Type here..."
                className="w-full rounded border p-2"
              />
              <button
                onClick={handleQuickAdd}
                className="rounded bg-blue-600 px-3 py-2 text-white transition hover:bg-blue-700"
              >
                Add
              </button>
            </div>
          </div>

          {/* 队友的任务列表 */}
          <TaskList />
          <Top3List />
        </div>

        {/* 右侧：周历视图 (你的部分) */}
        <div className="min-h-[600px] w-full rounded-xl border bg-white p-4 shadow-sm lg:w-2/3">
          <Calendar events={calendarEvents} />
        </div>
      </div>
    </div>
  );
}
