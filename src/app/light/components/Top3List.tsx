"use client";

import type { Task } from "../types";

type Props = {
  tasks: Task[];
  onRemove: (id: string) => void;  // 从top3移除
  onDelete: (id: string) => void;  // 删除任务
};

export default function Top3List({ tasks, onRemove, onDelete }: Props) {
  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold">⭐ Today Top 3</h2>

      {(!tasks || tasks.length === 0) && <p className="text-sm text-gray-500">暂未选择任务</p>}

      <div className="space-y-3">
        {tasks?.map((t) => (
          <div
            key={t.id}
            className="rounded-lg border px-3 py-2 text-gray-800"
            style={{ background: "linear-gradient(135deg,#f3f6ff,#fff)" }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{t.title}</div>
                <div className="text-xs text-gray-500">
                  {t.category === "deep" ? "深度任务" : t.category === "normal" ? "普通任务" : "快速任务"}
                  {t.estimatedTime ? ` · ${t.estimatedTime}m` : ""}
                </div>
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={() => onRemove(t.id)} 
                  className="rounded bg-yellow-50 px-2 py-1 text-xs text-yellow-700 hover:bg-yellow-100"
                >
                  Remove
                </button>
                <button 
                  onClick={() => onDelete(t.id)} 
                  className="rounded bg-red-50 px-2 py-1 text-xs text-red-700 hover:bg-red-100"
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
