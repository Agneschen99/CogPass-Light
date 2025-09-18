import React from "react";

const subjectColor: Record<string, string> = {
  math: "bg-blue-100 text-blue-700",
  history: "bg-amber-100 text-amber-700",
  english: "bg-emerald-100 text-emerald-700",
  cs: "bg-violet-100 text-violet-700",
};

export default function SubjectBadge({ subject }: { subject: string }) {
  const key = subject.toLowerCase();
  const cls = subjectColor[key] ?? "bg-slate-100 text-slate-700";
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {subject}
    </span>
  );
}
