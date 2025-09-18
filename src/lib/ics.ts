import dayjs from "dayjs";
import { Scheduled } from "@/types";

export function exportWeekToICS(slots: Array<Pick<Scheduled, "id"|"title"|"start"|"minutes">>) {
  let ics = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//NeuroPlan//EN\n";

  for (const s of slots) {
    const start = dayjs(s.start);
    const end = start.add(s.minutes, "minute");
    const fmt = "YYYYMMDDTHHmmss[Z]"; // UTC-ish

    ics += "BEGIN:VEVENT\n";
    ics += `UID:${s.id}\n`;
    ics += `DTSTAMP:${start.format(fmt)}\n`;
    ics += `DTSTART:${start.format(fmt)}\n`;
    ics += `DTEND:${end.format(fmt)}\n`;
    ics += `SUMMARY:${s.title}\n`;
    ics += "END:VEVENT\n";
  }

  ics += "END:VCALENDAR";
  return ics;
}
