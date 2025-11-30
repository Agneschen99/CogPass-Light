// src/components/Top3List.tsx
"use client";

import { useStore } from "@/state/store";

export default function Top3List() {
  const { todayTop3 } = useStore();

  return (
    <div
      style={{
        marginTop: 30,
        padding: 16,
        background: "#fff6e6",
        borderRadius: 10,
      }}
    >
      <h2 style={{ fontWeight: 700 }}>⭐ Today Top 3</h2>

      {todayTop3.length === 0 && <p style={{ marginTop: 10 }}>暂未选择任务</p>}

      {todayTop3.map((t: any) => (
        <div
          key={t.id}
          style={{
            background: "white",
            padding: 10,
            marginTop: 8,
            borderRadius: 6,
          }}
        >
          {t.title}
        </div>
      ))}
    </div>
  );
}
