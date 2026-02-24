"use client";

import type { Task } from "../types";

type Props = {
  tasks: Task[];
  onAddToTop3?: (taskId: string) => void;
  onDelete?: (taskId: string) => void;
};

function formatCategoryLabel(category: Task["category"]) {
  // 不展示给用户也行；如果你想完全隐藏，直接删掉调用处
  if (category === "deep") return "Deep";
  if (category === "normal") return "Normal";
  return "Quick";
}

export default function TaskBoard({ tasks, onAddToTop3, onDelete }: Props) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold">All Tasks</h3>
        <span className="text-xs text-gray-500">{tasks.length}</span>
      </div>

      <div className="flex flex-col gap-3">
        {tasks.length === 0 ? (
          <p className="text-sm text-gray-500">还没有任务～先在上面输入一个任务吧</p>
        ) : (
          tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-lg border border-gray-200 bg-white px-3 py-2"
            >
              <div className="flex items-center justify-between gap-3">
                <label className="flex items-center gap-2">
                  <input type="checkbox" />
                  <span className="text-gray-800">{task.title}</span>
                </label>

                <div className="flex items-center gap-2">
                  {/* 如果你想把类别完全对用户隐藏，把下面这一行删掉即可 */}
                  <span className="text-xs text-gray-400">
                    {formatCategoryLabel(task.category)}
                  </span>
                  {onAddToTop3 && (
                    <button
                      onClick={() => onAddToTop3(task.id)}
                      className="rounded bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100"
                    >
                      ⭐ Top3
                    </button>
                  )}
                  {onDelete && (
                    <button
                      onClick={() => onDelete(task.id)}
                      className="rounded bg-red-50 px-2 py-1 text-xs text-red-700 hover:bg-red-100"
                    >
                      🗑️
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-1 text-xs text-gray-500">
                {task.estimatedTime}m
                {task.dueDate ? ` · due ${task.dueDate}` : ""}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
