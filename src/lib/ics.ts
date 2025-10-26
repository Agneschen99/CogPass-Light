import dayjs from "dayjs";
import type { Slot } from "@/types";

export function exportWeekToICS(slots: Slot[] = []) {
  let ics = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//NeuroPlan//EN\n";

  for (const s of slots) {
    const start = dayjs(s.start);
    const minutes = (s as any).minutes ?? 25;
    const end = start.add(minutes, "minute");
    const fmt = "YYYYMMDDTHHmmss[Z]"; // UTC-ish

    ics += "BEGIN:VEVENT\n";
    ics += `UID:${(s as any).id ?? Math.random().toString(36).slice(2)}\n`;
    ics += `DTSTAMP:${start.format(fmt)}\n`;
    ics += `DTSTART:${start.format(fmt)}\n`;
    ics += `DTEND:${end.format(fmt)}\n`;
    ics += `SUMMARY:${(s.title ?? s.subject) ?? "Task"}\n`;
    ics += "END:VEVENT\n";
  }

  ics += "END:VCALENDAR";
  return ics;
}
