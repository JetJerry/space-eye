"""Automated experiment reporting system: HTML certification report, JSON summary, and CSV.

Part of VYOM Prompt 11: Experiment Reports & Cryptographic Verification Summaries.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vyom.audit.verifier import (
    AuditVerificationResult,
    AuditVerificationState,
    export_audit_manifest,
    verify_audit_chain,
)
from vyom.core.logging_setup import get_logger
from vyom.storage.models import RecordingRecord, SessionRecord

logger = get_logger("audit.reporter")


class ExperimentReporter:
    """Generates complete audit certification packages for experiment sessions."""

    def __init__(self, output_root: Path | str = "sessions") -> None:
        self.output_root = Path(output_root).resolve()

    def get_session_report_dir(self, session_id: str) -> Path:
        """Return dedicated reports folder for a given session."""
        report_dir = self.output_root / session_id / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def generate_csv_summary(
        self,
        session: SessionRecord,
        step_results: List[Dict[str, Any]],
        output_path: Path | str,
    ) -> Path:
        """Generate tabular step-by-step CSV summary."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "session_id",
            "step_order",
            "step_id",
            "step_name",
            "is_required",
            "state",
            "start_time_s",
            "completed_time_s",
            "elapsed_s",
            "evidence_duration_ms",
            "violations_count",
            "failure_reason",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in step_results:
                row = dict(r)
                row["session_id"] = session.session_id
                writer.writerow(row)

        logger.info("Generated CSV summary for '%s' at '%s'", session.session_id, path)
        return path

    def generate_json_report(
        self,
        session: SessionRecord,
        step_results: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        voice_alerts: List[Dict[str, Any]],
        recordings: List[RecordingRecord],
        verification_result: AuditVerificationResult,
        output_path: Path | str,
    ) -> Path:
        """Generate structured machine-readable JSON summary report."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "report_schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "session": session.to_dict(),
            "verification": verification_result.to_dict(),
            "metrics": {
                "total_steps": session.total_steps,
                "completed_steps": session.completed_steps,
                "completion_rate_pct": round(
                    (session.completed_steps / session.total_steps * 100) if session.total_steps > 0 else 0.0,
                    2,
                ),
                "total_violations": len(violations),
                "total_alerts": len(voice_alerts),
                "duration_seconds": round(session.duration_s, 2),
            },
            "step_results": step_results,
            "compliance_violations": violations,
            "voice_alerts": voice_alerts,
            "recordings": [r.to_dict() for r in recordings],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Generated JSON summary for '%s' at '%s'", session.session_id, path)
        return path

    def generate_html_report(
        self,
        session: SessionRecord,
        step_results: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        voice_alerts: List[Dict[str, Any]],
        recordings: List[RecordingRecord],
        verification_result: AuditVerificationResult,
        output_path: Path | str,
    ) -> Path:
        """Generate mission-control aerospace compliance certification HTML report."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Compute summary metrics
        completion_pct = (
            round((session.completed_steps / session.total_steps * 100), 1)
            if session.total_steps > 0
            else 0.0
        )
        is_verified = verification_result.is_valid and verification_result.status == AuditVerificationState.INTEGRITY_VERIFIED

        # Format rows for step results
        step_rows = []
        for s in step_results:
            st = s.get("state", "PENDING")
            badge_class = "badge-success" if st == "COMPLETED" else ("badge-danger" if st == "FAILED" else "badge-neutral")
            dur = f"{s.get('elapsed_s', 0.0):.1f}s" if s.get("elapsed_s") else "--"
            fail_msg = html.escape(str(s.get("failure_reason") or "--"))
            step_rows.append(
                f"""
                <tr>
                    <td class="text-center">{s.get('step_order', '--')}</td>
                    <td><code>{html.escape(str(s.get('step_id', '')))}</code></td>
                    <td><strong>{html.escape(str(s.get('step_name', '')))}</strong></td>
                    <td class="text-center"><span class="badge {badge_class}">{html.escape(st)}</span></td>
                    <td class="text-center">{'Yes' if s.get('is_required') else 'Optional'}</td>
                    <td class="text-center">{dur}</td>
                    <td class="text-muted">{fail_msg}</td>
                </tr>
                """
            )

        # Format rows for violations
        violation_rows = []
        if violations:
            for v in violations:
                sev = str(v.get("severity", "MEDIUM")).upper()
                sev_badge = "badge-danger" if sev in ("CRITICAL", "HIGH") else "badge-warning"
                time_str = f"{v.get('timestamp_s', 0.0):.2f}s"
                violation_rows.append(
                    f"""
                    <tr>
                        <td class="text-center">{time_str}</td>
                        <td class="text-center"><span class="badge {sev_badge}">{sev}</span></td>
                        <td><code>{html.escape(str(v.get('step_id') or '--'))}</code></td>
                        <td>{html.escape(str(v.get('violation_type', '')))}</td>
                        <td>{html.escape(str(v.get('explanation', '')))}</td>
                        <td class="text-muted"><em>"{html.escape(str(v.get('voice_message') or '--'))}"</em></td>
                    </tr>
                    """
                )
        else:
            violation_rows.append(
                """
                <tr>
                    <td colspan="6" class="text-center text-success" style="padding: 24px;">
                        ✓ Zero compliance violations recorded during this session.
                    </td>
                </tr>
                """
            )

        # Format recording rows
        rec_rows = []
        if recordings:
            for r in recordings:
                rec_rows.append(
                    f"""
                    <tr>
                        <td><code>{html.escape(Path(r.file_path).name)}</code></td>
                        <td class="text-center">{r.width}x{r.height}</td>
                        <td class="text-center">{r.fps:.1f} fps</td>
                        <td class="text-center">{r.duration_s:.1f}s</td>
                        <td class="text-center">{r.file_size_bytes / (1024*1024):.2f} MB</td>
                        <td><code class="hash-text">{r.sha256_checksum or 'Pending'}</code></td>
                    </tr>
                    """
                )
        else:
            rec_rows.append(
                """
                <tr>
                    <td colspan="6" class="text-center text-muted" style="padding: 18px;">
                        No video recording sessions registered.
                    </td>
                </tr>
                """
            )

        verification_badge = (
            '<span class="badge badge-success">✓ INTEGRITY VERIFIED</span>'
            if is_verified
            else f'<span class="badge badge-danger">✗ {html.escape(verification_result.status.value)}</span>'
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VYOM Experiment Certification: {html.escape(session.session_id)}</title>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-card: #121826;
            --bg-card-subtle: #182234;
            --border-color: #243048;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 32px;
        }}
        .title-group h1 {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
            margin-bottom: 6px;
        }}
        .title-group p {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        .verification-stamp {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 20px;
            text-align: right;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 18px 20px;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 32px;
            overflow: hidden;
        }}
        .section-header {{
            background: var(--bg-card-subtle);
            border-bottom: 1px solid var(--border-color);
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .section-header h2 {{
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-secondary);
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .text-center {{ text-align: center; }}
        .text-muted {{ color: var(--text-muted); }}
        .text-success {{ color: var(--accent-green); }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .badge-neutral {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid #94a3b8; }}
        code {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 5px;
            border-radius: 4px;
            color: #38bdf8;
        }}
        .hash-text {{
            word-break: break-all;
            font-size: 11px;
            color: #a78bfa;
        }}
        .crypto-box {{
            background: #080c14;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 16px;
            font-size: 12px;
            margin: 16px 20px;
        }}
        .crypto-row {{
            display: flex;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        .crypto-key {{
            width: 180px;
            color: var(--text-secondary);
            font-weight: 500;
            flex-shrink: 0;
        }}
        .crypto-val {{
            color: #e2e8f0;
            font-family: monospace;
            word-break: break-all;
        }}
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="title-group">
                <h1>VYOM Experiment Compliance Certification</h1>
                <p>Mission Session: <code>{html.escape(session.session_id)}</code> &bull; Protocol: <strong>{html.escape(session.protocol_name)}</strong> ({html.escape(session.protocol_id)} v{html.escape(session.protocol_version)})</p>
            </div>
            <div class="verification-stamp">
                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">AUDIT SEAL</div>
                {verification_badge}
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Session Status</div>
                <div class="kpi-value">{html.escape(session.status)}</div>
                <div class="kpi-sub">Duration: {session.duration_s:.1f}s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Step Progress</div>
                <div class="kpi-value">{session.completed_steps} / {session.total_steps}</div>
                <div class="kpi-sub">{completion_pct}% Execution Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Compliance Violations</div>
                <div class="kpi-value" style="color: {'var(--accent-green)' if len(violations) == 0 else 'var(--accent-red)'};">
                    {len(violations)}
                </div>
                <div class="kpi-sub">{len(voice_alerts)} Spoken Voice Alerts</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Audit Chain Events</div>
                <div class="kpi-value">{verification_result.total_records}</div>
                <div class="kpi-sub">SHA-256 Tamper-Evident Chain</div>
            </div>
        </div>

        <!-- Protocol Step Progression -->
        <div class="section">
            <div class="section-header">
                <h2>Procedure Execution Timeline</h2>
                <span class="text-muted" style="font-size: 12px;">{session.completed_steps} of {session.total_steps} steps completed</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th class="text-center" style="width: 50px;">#</th>
                        <th style="width: 140px;">Step ID</th>
                        <th>Step Description</th>
                        <th class="text-center" style="width: 120px;">Outcome</th>
                        <th class="text-center" style="width: 100px;">Required</th>
                        <th class="text-center" style="width: 90px;">Duration</th>
                        <th>Notes / Failure Explanation</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(step_rows)}
                </tbody>
            </table>
        </div>

        <!-- Compliance Violations -->
        <div class="section">
            <div class="section-header">
                <h2>Compliance Violations Log</h2>
                <span class="text-muted" style="font-size: 12px;">{len(violations)} incident(s) recorded</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th class="text-center" style="width: 80px;">Time</th>
                        <th class="text-center" style="width: 100px;">Severity</th>
                        <th style="width: 120px;">Step</th>
                        <th style="width: 160px;">Violation Type</th>
                        <th>Explanation</th>
                        <th style="width: 240px;">Spoken Voice Alert</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(violation_rows)}
                </tbody>
            </table>
        </div>

        <!-- Video Recordings & Session Artifacts -->
        <div class="section">
            <div class="section-header">
                <h2>Media Records & Checksums</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th class="text-center">Resolution</th>
                        <th class="text-center">Frame Rate</th>
                        <th class="text-center">Duration</th>
                        <th class="text-center">Size</th>
                        <th>SHA-256 Checksum</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rec_rows)}
                </tbody>
            </table>
        </div>

        <!-- Cryptographic Audit Chain Seal -->
        <div class="section">
            <div class="section-header">
                <h2>Cryptographic Audit Trail Summary</h2>
                <div>{verification_badge}</div>
            </div>
            <div class="crypto-box">
                <div class="crypto-row">
                    <div class="crypto-key">Audit Status:</div>
                    <div class="crypto-val">{html.escape(verification_result.status.value)}</div>
                </div>
                <div class="crypto-row">
                    <div class="crypto-key">Verified Records:</div>
                    <div class="crypto-val">{verification_result.total_records} events replayed & validated</div>
                </div>
                <div class="crypto-row">
                    <div class="crypto-key">Genesis Hash:</div>
                    <div class="crypto-val hash-text">{verification_result.genesis_hash or 'N/A'}</div>
                </div>
                <div class="crypto-row">
                    <div class="crypto-key">Final Root Digest:</div>
                    <div class="crypto-val hash-text"><strong>{verification_result.final_hash or 'N/A'}</strong></div>
                </div>
                <div class="crypto-row">
                    <div class="crypto-key">Verification Time (UTC):</div>
                    <div class="crypto-val">{verification_result.verification_timestamp_utc}</div>
                </div>
                <div class="crypto-row" style="margin-top: 12px; border-top: 1px dashed #1e293b; padding-top: 8px;">
                    <div class="crypto-key">Integrity Notice:</div>
                    <div class="crypto-val" style="color: var(--text-muted); font-family: sans-serif; font-size: 11px;">
                        Hash chaining provides strict mathematical tamper-evidence against retroactive modification, insertion, deletion, or reordering of internal experiment records. Verifiable authenticity is anchored by the detached cryptographic manifest digest.
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            Generated autonomously by VYOM (Offline Human Activity Recognition & Procedure Compliance System) &bull; SIH 26174
        </div>
    </div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Generated HTML certification report for '%s' at '%s'", session.session_id, path)
        return path

    def export_all_reports(
        self,
        session: SessionRecord,
        step_results: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        voice_alerts: List[Dict[str, Any]],
        recordings: List[RecordingRecord],
        verification_result: AuditVerificationResult,
        operator_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Generate and write all four standard session reports to disk."""
        rep_dir = self.get_session_report_dir(session.session_id)
        sid = session.session_id

        html_path = rep_dir / f"{sid}_report.html"
        json_path = rep_dir / f"{sid}_summary.json"
        csv_path = rep_dir / f"{sid}_steps.csv"
        manifest_path = rep_dir / f"{sid}_manifest.json"

        self.generate_html_report(
            session=session,
            step_results=step_results,
            violations=violations,
            voice_alerts=voice_alerts,
            recordings=recordings,
            verification_result=verification_result,
            output_path=html_path,
        )

        self.generate_json_report(
            session=session,
            step_results=step_results,
            violations=violations,
            voice_alerts=voice_alerts,
            recordings=recordings,
            verification_result=verification_result,
            output_path=json_path,
        )

        self.generate_csv_summary(
            session=session,
            step_results=step_results,
            output_path=csv_path,
        )

        recording_checksums = [
            {
                "file_path": r.file_path,
                "sha256": r.sha256_checksum,
                "file_size_bytes": r.file_size_bytes,
            }
            for r in recordings
        ]
        export_audit_manifest(
            session_id=session.session_id,
            protocol_id=session.protocol_id,
            protocol_version=session.protocol_version,
            verification_result=verification_result,
            output_path=manifest_path,
            operator_id=operator_id,
            recording_checksums=recording_checksums,
        )

        return {
            "html": html_path,
            "json": json_path,
            "csv": csv_path,
            "manifest": manifest_path,
        }

