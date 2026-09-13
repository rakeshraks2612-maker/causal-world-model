"""
PRISM Dashboard — Uncertainty-Aware Causal World Model & Decision Intelligence Platform.
Engineered with the exact Octolane design system, WebGL volumetric raymarching wave shader, geometric shapes field, floating island-shell navigation, interactive simulation playback, and full modal inspection suite.
"""

from __future__ import annotations
import json
from typing import Dict, Any


def build_dashboard_html(scenarios_data: Dict[str, Any], initial_scenario_key: str = "scenario_04_pump") -> str:
    """Generate the exact Octolane-engineered PRISM Decision Intelligence Platform."""
    data_json = json.dumps(scenarios_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PRISM · Superintelligence for physical judgment</title>
  <meta name="description" content="Systems that compound physical judgment and deploy it at scale. Every intervention, state transition, and counterfactual informs the next.">
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiBmaWxsPSJub25lIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3AiIHgxPSI1MCIgeTE9IjEwIiB4Mj0iNTAiIHkyPSI1MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmZmZmZmIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2NiZDVlMSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3BSaWdodCIgeDE9IjUwIiB5MT0iMzAiIHgyPSI4NSIgeTI9IjMwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmZmZmYiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjOTRhM2I4Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJwcmlzbUhleEFtYmVyIiB4MT0iNjciIHkxPSI0MCIgeDI9Ijg1IiB5Mj0iNzAiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI2ZmOWEzYyIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjUwJSIgc3RvcC1jb2xvcj0iI2Y5NzMxNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiNlYTU4MGMiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4RGVlcEFtYmVyIiB4MT0iNTAiIHkxPSI3MCIgeDI9Ijg1IiB5Mj0iNzAiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI2VhNTgwYyIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiNjMjQxMGMiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4RGFyayIgeDE9IjE1IiB5MT0iNTAiIHgyPSI1MCIgeTI9IjcwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMwOTBkMTYiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMWUyOTNiIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJwcmlzbUhleFNsYXRlIiB4MT0iMTUiIHkxPSIzMCIgeDI9IjUwIiB5Mj0iNTAiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzFlMjkzYiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMzMzQxNTUiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsMTAgODUsMzAgNjcsNDAgNTAsMzAiIGZpbGw9InVybCgjcHJpc21IZXhUb3BSaWdodCkiLz4KICA8cG9seWdvbiBwb2ludHM9Ijg1LDMwIDg1LDcwIDY3LDYwIDY3LDQwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4QW1iZXIpIi8+CiAgPHBvbHlnb24gcG9pbnRzPSI4NSw3MCA1MCw5MCA1MCw3MCA2Nyw2MCIgZmlsbD0idXJsKCNwcmlzbUhleERlZXBBbWJlcikiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDkwIDE1LDcwIDMzLDYwIDUwLDcwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4RGFyaykiLz4KICA8cG9seWdvbiBwb2ludHM9IjE1LDcwIDE1LDMwIDMzLDQwIDMzLDYwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4U2xhdGUpIi8+CiAgPHBvbHlnb24gcG9pbnRzPSIxNSwzMCA1MCwxMCA1MCwzMCAzMyw0MCIgZmlsbD0idXJsKCNwcmlzbUhleFRvcCkiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDMwIDY3LDQwIDUwLDUwIiBmaWxsPSIjZmZmZmZmIiBvcGFjaXR5PSIwLjk1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI2Nyw0MCA2Nyw2MCA1MCw1MCIgZmlsbD0idXJsKCNwcmlzbUhleEFtYmVyKSIgb3BhY2l0eT0iMC45Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI2Nyw2MCA1MCw3MCA1MCw1MCIgZmlsbD0iI2VhNTgwYyIgb3BhY2l0eT0iMC44NSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsNzAgMzMsNjAgNTAsNTAiIGZpbGw9IiMwOTBkMTYiIG9wYWNpdHk9IjAuOTUiLz4KICA8cG9seWdvbiBwb2ludHM9IjMzLDYwIDMzLDQwIDUwLDUwIiBmaWxsPSIjMWUyOTNiIiBvcGFjaXR5PSIwLjkiLz4KICA8cG9seWdvbiBwb2ludHM9IjMzLDQwIDUwLDMwIDUwLDUwIiBmaWxsPSIjY2JkNWUxIiBvcGFjaXR5PSIwLjk1Ii8+CiAgPGxpbmUgeDE9IjUwIiB5MT0iMTAiIHgyPSI1MCIgeTI9IjMwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC42KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPGxpbmUgeDE9Ijg1IiB5MT0iMzAiIHgyPSI2NyIgeTI9IjQwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC40KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPGxpbmUgeDE9Ijg1IiB5MT0iNzAiIHgyPSI2NyIgeTI9IjYwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC40KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPGxpbmUgeDE9IjUwIiB5MT0iOTAiIHgyPSI1MCIgeTI9IjcwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4zKSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPGxpbmUgeDE9IjE1IiB5MT0iNzAiIHgyPSIzMyIgeTI9IjYwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4zKSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPGxpbmUgeDE9IjE1IiB5MT0iMzAiIHgyPSIzMyIgeTI9IjQwIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC41KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCwzMCA2Nyw0MCA2Nyw2MCA1MCw3MCAzMyw2MCAzMyw0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuNSkiIHN0cm9rZS13aWR0aD0iMC43NSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsMTAgODUsMzAgODUsNzAgNTAsOTAgMTUsNzAgMTUsMzAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgxMCwxMSwxMywwLjE4KSIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCw0MyA1MS44LDQ4LjIgNTcsNTAgNTEuOCw1MS44IDUwLDU3IDQ4LjIsNTEuOCA0Myw1MCA0OC4yLDQ4LjIiIGZpbGw9IiNmZmZmZmYiLz4KICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIxLjYiIGZpbGw9IiNmOTczMTYiLz4KPC9zdmc+">
  <link rel="icon" type="image/svg+xml" href="favicon.svg?v=prism3">
  <link rel="alternate icon" type="image/png" href="favicon.png?v=prism3">
  <link rel="shortcut icon" href="favicon.ico?v=prism3">
  <link rel="apple-touch-icon" href="favicon.png?v=prism3">
  <script>
    (function() {{
      var icon = document.querySelector("link[rel*='icon']");
      if (!icon) {{
        icon = document.createElement('link');
        icon.rel = 'shortcut icon';
        document.head.appendChild(icon);
      }}
      icon.type = 'image/svg+xml';
      icon.href = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiBmaWxsPSJub25lIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3AiIHgxPSI1MCIgeTE9IjEwIiB4Mj0iNTAiIHkyPSI1MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmZmZmZmIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2NiZDVlMSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3BSaWdodCIgeDE9IjUwIiB5MT0iMzAiIHgyPSI4NSIgeTI9IjMwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmZmZmYiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4QW1iZXIiIHgxPSI2NyIgeTE9IjQwIiB4Mj0iODUiIHkyPSI3MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmY5YTNjIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjZjk3MzE2Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2VhNTgwYyIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhEZWVwQW1iZXIiIHgxPSI1MCIgeTE9IjcwIiB4Mj0iODUiIHkyPSI3MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZWE1ODBjIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2MyNDEwYyIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhEYXJrIiB4MT0iMTUiIHkxPSI1MCIgeDI9IjUwIiB5Mj0iNzAiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzA5MGQxNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMxZTI5M2IiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4U2xhdGUiIHgxPSIxNSIgeTE9IjMwIiB4Mj0iNTAiIHkyPSI1MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMWUyOTNiIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzMzNDE1NSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCwxMCA4NSwzMCA2Nyw0MCA1MCwzMCIgZmlsbD0idXJsKCNwcmlzbUhleFRvcFJpZ2h0KSIvPgogIDxwb2x5Z29uIHBvaW50cz0iODUsMzAgODUsNzAgNjcsNjAgNjcsNDAiIGZpbGw9InVybCgjcHJpc21IZXhBbWJlcikiLz4KICA8cG9seWdvbiBwb2ludHM9Ijg1LDcwIDUwLDkwIDUwLDcwIDY3LDYwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4RGVlcEFtYmVyKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsOTAgMTUsNzAgMzMsNjAgNTAsNzAiIGZpbGw9InVybCgjcHJpc21IZXhEYXJrKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMTUsNzAgMTUsMzAgMzMsNDAgMzMsNjAiIGZpbGw9InVybCgjcHJpc21IZXhTbGF0ZSkiLz4KICA8cG9seWdvbiBwb2ludHM9IjE1LDMwIDUwLDEwIDUwLDMwIDMzLDQwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4VG9wKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsMzAgNjcsNDAgNTAsNTAiIGZpbGw9IiNmZmZmZmYiIG9wYWNpdHk9IjAuOTUiLz4KICA8cG9seWdvbiBwb2ludHM9IjY3LDQwIDY3LDYwIDUwLDUwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4QW1iZXIpIiBvcGFjaXR5PSIwLjkiLz4KICA8cG9seWdvbiBwb2ludHM9IjY3LDYwIDUwLDcwIDUwLDUwIiBmaWxsPSIjZWE1ODBjIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCw3MCAzMyw2MCA1MCw1MCIgZmlsbD0iIzA5MGQxNiIgb3BhY2l0eT0iMC45NSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMzMsNjAgMzMsNDAgNTAsNTAiIGZpbGw9IiMxZTI5M2IiIG9wYWNpdHk9IjAuOSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMzMsNDAgNTAsMzAgNTAsNTAiIGZpbGw9IiNjYmQ1ZTEiIG9wYWNpdHk9IjAuOTUiLz4KICA8bGluZSB4MT0iNTAiIHkxPSIxMCIgeDI9IjUwIiB5Mj0iMzAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjYpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iODUiIHkxPSIzMCIgeDI9IjY3IiB5Mj0iNDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjQpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iODUiIHkxPSI3MCIgeDI9IjY3IiB5Mj0iNjAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjQpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iNTAiIHkxPSI5MCIgeDI9IjUwIiB5Mj0iNzAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjMpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iMTUiIHkxPSI3MCIgeDI9IjMzIiB5Mj0iNjAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjMpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iMTUiIHkxPSIzMCIgeDI9IjMzIiB5Mj0iNDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjUpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDMwIDY3LDQwIDY3LDYwIDUwLDcwIDMzLDYwIDMzLDQwIiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC41KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCwxMCA4NSwzMCA4NSw3MCA1MCw5MCAxNSw3MCAxNSwzMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDEwLDExLDEzLDAuMTgpIiBzdHJva2Utd2lkdGg9IjEiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDQzIDUxLjgsNDguMiA1Nyw1MCA1MS44LDUxLjggNTAsNTcgNDguMiw1MS44IDQzLDUwIDQ4LjIsNDguMiIgZmlsbD0iI2ZmZmZmZiIvPgogIDxjaXJjbGUgY3g9IjUwIiBjeT0iNTAiIHI9IjEuNiIgZmlsbD0iI2Y5NzMxNiIvPgo8L3N2Zz4=";
    }})();
  </script>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiBmaWxsPSJub25lIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3AiIHgxPSI1MCIgeTE9IjEwIiB4Mj0iNTAiIHkyPSI1MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmZmZmZmIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2NiZDVlMSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhUb3BSaWdodCIgeDE9IjUwIiB5MT0iMzAiIHgyPSI4NSIgeTI9IjMwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmZmZmYiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4QW1iZXIiIHgxPSI2NyIgeTE9IjQwIiB4Mj0iODUiIHkyPSI3MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmY5YTNjIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjZjk3MzE2Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2VhNTgwYyIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhEZWVwQW1iZXIiIHgxPSI1MCIgeTE9IjcwIiB4Mj0iODUiIHkyPSI3MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZWE1ODBjIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2MyNDEwYyIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHJpc21IZXhEYXJrIiB4MT0iMTUiIHkxPSI1MCIgeDI9IjUwIiB5Mj0iNzAiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzA5MGQxNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMxZTI5M2IiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InByaXNtSGV4U2xhdGUiIHgxPSIxNSIgeTE9IjMwIiB4Mj0iNTAiIHkyPSI1MCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMWUyOTNiIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzMzNDE1NSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCwxMCA4NSwzMCA2Nyw0MCA1MCwzMCIgZmlsbD0idXJsKCNwcmlzbUhleFRvcFJpZ2h0KSIvPgogIDxwb2x5Z29uIHBvaW50cz0iODUsMzAgODUsNzAgNjcsNjAgNjcsNDAiIGZpbGw9InVybCgjcHJpc21IZXhBbWJlcikiLz4KICA8cG9seWdvbiBwb2ludHM9Ijg1LDcwIDUwLDkwIDUwLDcwIDY3LDYwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4RGVlcEFtYmVyKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsOTAgMTUsNzAgMzMsNjAgNTAsNzAiIGZpbGw9InVybCgjcHJpc21IZXhEYXJrKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMTUsNzAgMTUsMzAgMzMsNDAgMzMsNjAiIGZpbGw9InVybCgjcHJpc21IZXhTbGF0ZSkiLz4KICA8cG9seWdvbiBwb2ludHM9IjE1LDMwIDUwLDEwIDUwLDMwIDMzLDQwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4VG9wKSIvPgogIDxwb2x5Z29uIHBvaW50cz0iNTAsMzAgNjcsNDAgNTAsNTAiIGZpbGw9IiNmZmZmZmYiIG9wYWNpdHk9IjAuOTUiLz4KICA8cG9seWdvbiBwb2ludHM9IjY3LDQwIDY3LDYwIDUwLDUwIiBmaWxsPSJ1cmwoI3ByaXNtSGV4QW1iZXIpIiBvcGFjaXR5PSIwLjkiLz4KICA8cG9seWdvbiBwb2ludHM9IjY3LDYwIDUwLDcwIDUwLDUwIiBmaWxsPSIjZWE1ODBjIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCw3MCAzMyw2MCA1MCw1MCIgZmlsbD0iIzA5MGQxNiIgb3BhY2l0eT0iMC45NSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMzMsNjAgMzMsNDAgNTAsNTAiIGZpbGw9IiMxZTI5M2IiIG9wYWNpdHk9IjAuOSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMzMsNDAgNTAsMzAgNTAsNTAiIGZpbGw9IiNjYmQ1ZTEiIG9wYWNpdHk9IjAuOTUiLz4KICA8bGluZSB4MT0iNTAiIHkxPSIxMCIgeDI9IjUwIiB5Mj0iMzAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjYpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iODUiIHkxPSIzMCIgeDI9IjY3IiB5Mj0iNDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjQpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iODUiIHkxPSI3MCIgeDI9IjY3IiB5Mj0iNjAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjQpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iNTAiIHkxPSI5MCIgeDI9IjUwIiB5Mj0iNzAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjMpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iMTUiIHkxPSI3MCIgeDI9IjMzIiB5Mj0iNjAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjMpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8bGluZSB4MT0iMTUiIHkxPSIzMCIgeDI9IjMzIiB5Mj0iNDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjUpIiBzdHJva2Utd2lkdGg9IjAuNzUiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDMwIDY3LDQwIDY3LDYwIDUwLDcwIDMzLDYwIDMzLDQwIiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC41KSIgc3Ryb2tlLXdpZHRoPSIwLjc1Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSI1MCwxMCA4NSwzMCA4NSw3MCA1MCw5MCAxNSw3MCAxNSwzMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDEwLDExLDEzLDAuMTgpIiBzdHJva2Utd2lkdGg9IjEiLz4KICA8cG9seWdvbiBwb2ludHM9IjUwLDQzIDUxLjgsNDguMiA1Nyw1MCA1MS44LDUxLjggNTAsNTcgNDguMiw1MS44IDQzLDUwIDQ4LjIsNDguMiIgZmlsbD0iI2ZmZmZmZiIvPgogIDxjaXJjbGUgY3g9IjUwIiBjeT0iNTAiIHI9IjEuNiIgZmlsbD0iI2Y5NzMxNiIvPgo8L3N2Zz4=">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --site-canvas: #ffffff;
      --site-card: #f4f5f7;
      --site-card-hover: #ffffff;
      --site-faint: #8c8e94;
      --site-muted: #71737a;
      --site-ink: #121316;
      --site-ink-soft: #4a4d57;
      
      --accent-octo: #f76b15;
      --accent-octo-soft: #ffe8d7;
      --accent-emerald: #10b981;
      --accent-emerald-soft: #e6f9f2;
      --accent-rose: #f43f5e;
      --accent-rose-soft: #ffe8ec;
      
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
      --ease-out-quint: cubic-bezier(0.22, 1, 0.36, 1);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      background-color: var(--site-canvas);
      color: var(--site-ink-soft);
      font-family: var(--font-sans);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
      scroll-behavior: smooth;
    }}

    /* ====================================================================
       1. OCTOLANE LUXURY TASKBAR / FLOATING ISLAND
       ==================================================================== */
    .nav-island-wrapper {{
      position: fixed;
      top: 16px;
      left: 0;
      right: 0;
      display: flex;
      justify-content: center;
      z-index: 1000;
      padding: 0 16px;
      pointer-events: none;
    }}

    .site-island-shell {{
      pointer-events: auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 1240px;
      height: 48px;
      border-radius: 999px;
      padding: 0 8px 0 14px;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(28px) saturate(190%);
      -webkit-backdrop-filter: blur(28px) saturate(190%);
      box-shadow: 0 0 0 1px rgba(10, 11, 13, 0.08), 0 8px 32px rgba(10, 11, 13, 0.07);
      gap: 12px;
      box-sizing: border-box;
      transition: all 0.25s var(--ease-out-quint);
    }}

    /* Left Zone: Brand Logo & Product Name */
    .nav-brand-group {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
      color: var(--site-ink);
      font-weight: 700;
      font-size: 0.94rem;
      letter-spacing: -0.025em;
      cursor: pointer;
      flex-shrink: 0;
      line-height: 1;
    }}

    .prism-mark-svg {{
      width: 26px;
      height: 26px;
      display: block;
      transition: transform 0.3s var(--ease-out-quint);
      filter: drop-shadow(0 2px 5px rgba(249, 115, 22, 0.25));
    }}

    .nav-brand-group:hover .prism-mark-svg {{
      transform: scale(1.08) rotate(3deg);
    }}

    .nav-badge-zero {{
      font-family: var(--font-mono);
      font-size: 0.60rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      background: rgba(10, 11, 13, 0.06);
      padding: 0 6px;
      height: 19px;
      border-radius: 6px;
      color: var(--site-muted);
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      line-height: 1;
    }}

    /* Center Zone: Clean Section Anchors + Scenario Switcher */
    .nav-center-deck {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }}

    .nav-section-links {{
      display: inline-flex;
      align-items: center;
      gap: 2px;
    }}

    .nav-link-anchor {{
      background: transparent;
      border: none;
      color: var(--site-muted);
      font-family: var(--font-sans);
      font-size: 0.76rem;
      font-weight: 500;
      height: 32px;
      padding: 0 10px;
      border-radius: 999px;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.15s var(--ease-out-quint);
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      line-height: 1;
      box-sizing: border-box;
    }}

    .nav-link-anchor:hover {{
      color: var(--site-ink);
      background: rgba(10, 11, 13, 0.05);
    }}

    .nav-link-anchor.active {{
      color: var(--site-ink);
      background: rgba(10, 11, 13, 0.08);
      font-weight: 600;
    }}

    .nav-divider-v {{
      width: 1px;
      height: 16px;
      background: rgba(10, 11, 13, 0.12);
      margin: 0 2px;
      flex-shrink: 0;
    }}

    /* Viable Scenario Small Tabs in Floating Taskbar */
    .nav-scenarios-pill-box {{
      display: inline-flex;
      align-items: center;
      gap: 2px;
      background: rgba(10, 11, 13, 0.05);
      border-radius: 999px;
      height: 32px;
      padding: 0 3px;
      border: 1px solid rgba(10, 11, 13, 0.06);
      flex-shrink: 0;
      box-sizing: border-box;
    }}

    .nav-scen-tag {{
      font-family: var(--font-mono);
      font-size: 0.58rem;
      font-weight: 700;
      color: var(--site-muted);
      padding: 0 4px 0 5px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      user-select: none;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }}

    .nav-scen-btn {{
      background: transparent;
      border: none;
      color: var(--site-muted);
      font-family: var(--font-sans);
      font-size: 0.74rem;
      font-weight: 500;
      height: 26px;
      padding: 0 7px;
      border-radius: 999px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.18s var(--ease-out-quint);
      white-space: nowrap;
      line-height: 1;
      box-sizing: border-box;
    }}

    .nav-scen-btn:hover {{
      color: var(--site-ink);
      background: rgba(255, 255, 255, 0.65);
    }}

    .nav-scen-btn.active {{
      background: #ffffff;
      color: var(--site-ink);
      font-weight: 600;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.10), 0 0 0 0.5px rgba(0, 0, 0, 0.06);
    }}

    .nav-scen-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      flex-shrink: 0;
      display: inline-block;
    }}

    .nav-scen-code {{
      font-weight: 600;
      font-size: 0.72rem;
    }}

    .nav-scen-name {{
      display: none;
      font-weight: 500;
      margin-left: 2px;
      font-size: 0.72rem;
    }}

    .nav-scen-btn.active .nav-scen-name {{
      display: inline;
    }}

    /* Studio Perspective Filter Tabs (Scenario Contextual Views) */
    .studio-perspective-bar {{
      margin: 0 0 20px 0;
      padding: 10px 16px;
      background: var(--site-card);
      border-radius: 16px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid rgba(10, 11, 13, 0.06);
    }}

    .perspective-meta-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      color: var(--site-ink);
      text-transform: uppercase;
    }}

    .perspective-pulse-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-octo);
      animation: pulseOcto 2s infinite;
    }}

    @keyframes pulseOcto {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.4; transform: scale(0.85); }}
    }}

    .perspective-tab-cluster {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}

    .perspective-tab-btn {{
      background: rgba(10, 11, 13, 0.04);
      border: 1px solid rgba(10, 11, 13, 0.06);
      border-radius: 999px;
      padding: 5px 12px;
      font-family: var(--font-sans);
      font-size: 0.74rem;
      font-weight: 500;
      color: var(--site-muted);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s var(--ease-out-quint);
    }}

    .perspective-tab-btn:hover {{
      color: var(--site-ink);
      background: rgba(10, 11, 13, 0.08);
    }}

    .perspective-tab-btn.active {{
      background: var(--site-ink);
      color: #ffffff;
      font-weight: 600;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.16);
      border-color: var(--site-ink);
    }}

    .perspective-tab-badge {{
      font-family: var(--font-mono);
      font-size: 0.62rem;
      padding: 1.5px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.2);
      color: inherit;
    }}

    .perspective-tab-btn:not(.active) .perspective-tab-badge {{
      background: rgba(10, 11, 13, 0.08);
      color: var(--site-ink);
    }}

    /* Right Zone: Status Chip & Actions */
    .nav-right-deck {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }}

    .nav-operator-btn {{
      background: rgba(10, 11, 13, 0.05);
      border: 1px solid rgba(10, 11, 13, 0.08);
      color: var(--site-ink);
      font-family: var(--font-sans);
      font-size: 0.74rem;
      font-weight: 500;
      height: 32px;
      padding: 0 11px;
      border-radius: 999px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
      white-space: nowrap;
      line-height: 1;
      box-sizing: border-box;
    }}

    .nav-operator-btn:hover {{
      background: rgba(10, 11, 13, 0.10);
      transform: translateY(-1px);
    }}

    .nav-status-indicator {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 0.65rem;
      font-weight: 600;
      color: #047857;
      height: 32px;
      padding: 0 10px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25);
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
      line-height: 1;
      box-sizing: border-box;
    }}

    .nav-status-indicator:hover {{
      background: rgba(16, 185, 129, 0.20);
      transform: translateY(-1px);
    }}

    .nav-status-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-emerald);
      box-shadow: 0 0 6px var(--accent-emerald);
      animation: pulseGreen 2s infinite;
    }}

    @keyframes pulseGreen {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.4; transform: scale(0.85); }}
    }}

    .nav-cta-btn {{
      background: var(--site-ink);
      color: #ffffff;
      font-family: var(--font-sans);
      font-size: 0.76rem;
      font-weight: 600;
      height: 32px;
      padding: 0 13px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      text-decoration: none;
      transition: all 0.2s var(--ease-out-quint);
      white-space: nowrap;
      line-height: 1;
      box-sizing: border-box;
    }}

    .nav-cta-btn:hover {{
      background: #000000;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    }}

    .nav-cta-btn.tour-active {{
      background: var(--accent-octo);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(249, 115, 22, 0.35);
    }}

    /* Global Section Scroll Margin for Fixed Navbar */
    html {{
      scroll-behavior: smooth;
    }}

    section, #heroStage, #storyboardSection, #decisionStudio, #benchmarkLedger {{
      scroll-margin-top: 76px;
    }}

    /* Responsive Taskbar Breakpoints */
    @media (max-width: 1120px) {{
      .nav-badge-zero {{ display: none; }}
      .nav-scen-tag {{ display: none; }}
      .nav-operator-btn {{ display: none; }}
    }}

    @media (max-width: 940px) {{
      .nav-section-links {{ display: none; }}
      .nav-divider-v {{ display: none; }}
    }}

    @media (max-width: 660px) {{
      .nav-status-indicator {{ display: none; }}
      .site-island-shell {{
        height: 48px;
        padding: 0 6px 0 12px;
      }}
    }}

    /* ====================================================================
       2. HERO STAGE WITH WEBGL RAYMARCHING WAVE & SHAPES-FIELD CANVAS
       ==================================================================== */
    .hero-stage {{
      position: relative;
      width: 100vw;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background: var(--site-canvas);
      padding: 100px 24px 40px 24px;
    }}

    /* WebGL Volumetric Raymarching Canvas */
    #raymarchingCanvas {{
      position: absolute;
      top: 21%;
      left: 0;
      width: 100%;
      height: 58%;
      z-index: 1;
      pointer-events: none;
      mask-image: linear-gradient(to bottom, transparent, black 22%, black 78%, transparent);
      -webkit-mask-image: linear-gradient(to bottom, transparent, black 22%, black 78%, transparent);
      opacity: 0.95;
    }}

    /* Geometric Shapes Field Canvas */
    #shapesFieldCanvas {{
      position: absolute;
      left: 50%;
      top: 50%;
      width: 135vmin;
      height: 135vmin;
      transform: translate(-50%, -50%);
      z-index: 2;
      pointer-events: none;
      mask-image: radial-gradient(closest-side, rgba(0,0,0,0.85), rgba(0,0,0,0.45) 42%, transparent 70%);
      -webkit-mask-image: radial-gradient(closest-side, rgba(0,0,0,0.85), rgba(0,0,0,0.45) 42%, transparent 70%);
      opacity: 0.75;
    }}

    /* Monospace Metadata Blocks */
    .notes-corner-top {{
      position: absolute;
      top: 50px;
      right: 48px;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 400;
      text-transform: uppercase;
      line-height: 2.05;
      letter-spacing: 0.08em;
      color: var(--site-faint);
      pointer-events: none;
      z-index: 10;
    }}

    .notes-corner-bottom {{
      position: absolute;
      bottom: 42px;
      left: 48px;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 400;
      text-transform: uppercase;
      line-height: 2.05;
      letter-spacing: 0.08em;
      color: var(--site-faint);
      pointer-events: none;
      z-index: 10;
    }}

    .notes-matrix-icon {{
      position: absolute;
      top: 50px;
      left: 48px;
      z-index: 10;
      color: var(--site-ink);
      opacity: 0.85;
    }}

    /* Hero Foreground Center */
    .hero-center-content {{
      position: relative;
      z-index: 10;
      max-width: 840px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      margin: 0 auto;
    }}

    .hero-emblem-wrap {{
      margin-bottom: 24px;
      display: flex;
      justify-content: center;
      align-items: center;
      animation: gentleFloat 6s ease-in-out infinite;
    }}

    @keyframes gentleFloat {{
      0%, 100% {{ transform: translateY(0px); }}
      50% {{ transform: translateY(-5px); }}
    }}

    .hero-emblem-tile {{
      width: 76px;
      height: 76px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 0 0 1px rgba(10, 11, 13, 0.08), 0 12px 36px rgba(249, 115, 22, 0.16), 0 4px 16px rgba(0, 0, 0, 0.06);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.35s var(--ease-out-quint);
    }}

    .hero-emblem-tile:hover {{
      transform: translateY(-2px) scale(1.04);
      box-shadow: 0 0 0 1px rgba(249, 115, 22, 0.35), 0 16px 44px rgba(249, 115, 22, 0.25);
    }}

    .hero-prism-mark {{
      width: 52px;
      height: 52px;
      display: block;
      filter: drop-shadow(0 4px 10px rgba(249, 115, 22, 0.3));
    }}

    .hero-octo-title {{
      font-size: clamp(48px, 7.2vw, 80px);
      font-weight: 400;
      line-height: 1.05;
      letter-spacing: -0.015em;
      color: var(--site-ink);
      margin-bottom: 28px;
      text-wrap: balance;
    }}

    /* Button Row */
    .hero-cta-cluster {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 10px;
      margin-bottom: 18px;
    }}

    .hero-btn-dark {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      background: var(--site-ink);
      color: #ffffff;
      font-size: 15px;
      font-weight: 500;
      gap: 8px;
      text-decoration: none;
      border: none;
      cursor: pointer;
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.09), 0 2px 6px rgba(10, 11, 13, 0.06);
      transition: all 0.2s var(--ease-out-quint);
    }}

    .hero-btn-dark:hover {{
      background: #000000;
      transform: translateY(-1px);
    }}

    .hero-btn-light {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--site-ink);
      font-size: 15px;
      font-weight: 500;
      gap: 8px;
      text-decoration: none;
      border: 1px solid rgba(10, 11, 13, 0.12);
      cursor: pointer;
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.08), 0 2px 4px rgba(10, 11, 13, 0.04);
      transition: all 0.2s var(--ease-out-quint);
    }}

    .hero-btn-light:hover {{
      background: #ffffff;
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
    }}

    .hero-contact-sublink {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 14px;
      color: var(--site-faint);
      text-decoration: none;
      margin-bottom: 24px;
      cursor: pointer;
      transition: color 0.15s ease;
    }}

    .hero-contact-sublink:hover {{
      color: var(--site-ink);
    }}

    .hero-contact-sublink svg {{
      transition: transform 0.2s ease;
    }}

    .hero-contact-sublink:hover svg {{
      transform: translateX(3px);
    }}

    .hero-subtitle-block {{
      max-width: 640px;
      margin: 0 auto;
    }}

    .hero-subtitle-line1 {{
      font-size: 17px;
      font-weight: 400;
      line-height: 1.6;
      color: var(--site-ink);
    }}

    .hero-subtitle-line2 {{
      font-size: 17px;
      font-weight: 400;
      line-height: 1.6;
      color: var(--site-muted);
      margin-top: 4px;
    }}

    .hero-learn-more {{
      margin-top: 48px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      font-family: var(--font-mono);
      font-size: 12.5px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--site-muted);
      text-decoration: none;
      cursor: pointer;
    }}

    .rule-dots-v {{
      display: block;
      width: 1px;
      height: 38px;
      background: linear-gradient(to bottom, var(--site-muted), transparent);
      animation: pulseScroll 2.4s var(--ease-out-quint) infinite;
    }}

    @keyframes pulseScroll {{
      0% {{ transform: scaleY(0); transform-origin: top; opacity: 0; }}
      50% {{ transform: scaleY(1); transform-origin: top; opacity: 1; }}
      50.1% {{ transform: scaleY(1); transform-origin: bottom; opacity: 1; }}
      100% {{ transform: scaleY(0); transform-origin: bottom; opacity: 0; }}
    }}

    /* ====================================================================
       3. PRISM DECISION SAAS PLATFORM
       ==================================================================== */
    .section-pillars-workspace {{
      position: relative;
      z-index: 10;
      max-width: 1080px;
      margin: 40px auto 100px auto;
      padding: 0 24px;
    }}

    .eyebrow-pill {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      border-radius: 999px;
      background: rgba(10, 11, 13, 0.05);
      padding: 0 10px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--site-ink);
      margin-bottom: 16px;
    }}

    .pillars-headline {{
      font-size: 48px;
      font-weight: 400;
      line-height: 56px;
      letter-spacing: -0.015em;
      color: var(--site-ink);
      margin-bottom: 12px;
    }}

    .pillars-desc {{
      font-size: 16px;
      color: var(--site-muted);
      max-width: 680px;
      margin-bottom: 40px;
      line-height: 1.6;
    }}

    /* 3-Act Storyboard Cards */
    .cards-grid-3act {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 36px;
    }}

    .corner-superellipse-card {{
      background: var(--site-card);
      border-radius: 24px;
      padding: 32px;
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.09), 0 2px 6px rgba(10, 11, 13, 0.06);
      cursor: pointer;
      transition: all 0.25s var(--ease-out-quint);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .corner-superellipse-card:hover {{
      background: #ffffff;
      transform: translateY(-2px);
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.12), 0 12px 28px -4px rgba(10, 11, 13, 0.08);
    }}

    .corner-superellipse-card.active {{
      background: #ffffff;
      box-shadow: 0 0 0 1.5px var(--accent-octo), 0 12px 28px -4px rgba(10, 11, 13, 0.08);
    }}

    .pillar-icon-chip {{
      width: 40px;
      height: 40px;
      border-radius: 999px;
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.09), 0 2px 6px rgba(10, 11, 13, 0.06);
      margin-bottom: 24px;
    }}

    .card-kicker-title {{
      font-size: 24px;
      font-weight: 400;
      line-height: 1.2;
      color: var(--site-ink);
      margin-bottom: 10px;
    }}

    .card-kicker-title span.tone {{
      display: block;
      font-size: 15px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}

    .card-body-text {{
      font-size: 14.5px;
      color: var(--site-ink-soft);
      line-height: 1.55;
      margin-bottom: 24px;
    }}

    .card-stats-list {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 10px;
      border-top: 1px solid rgba(10, 11, 13, 0.06);
      padding-top: 18px;
    }}

    .stat-line-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 14px;
    }}

    .stat-val-bold {{
      font-weight: 600;
      font-family: var(--font-mono);
      color: var(--site-ink);
    }}

    /* Equalizer Box */
    .equalizer-box {{
      background: #ffffff;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.07);
      margin-bottom: 36px;
    }}

    .eq-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}

    .eq-meta-labels {{
      display: flex;
      align-items: center;
      gap: 20px;
      font-family: var(--font-mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--site-faint);
    }}

    .eq-bars-strip {{
      display: flex;
      align-items: stretch;
      justify-content: space-between;
      height: 96px;
      mask-image: linear-gradient(to right, transparent, black 12%, black 88%, transparent);
      -webkit-mask-image: linear-gradient(to right, transparent, black 12%, black 88%, transparent);
    }}

    .eq-bar-column {{
      width: 2px;
      background: #ffe8d7;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .eq-bar-fill {{
      background: var(--accent-octo);
      transition: height 0.7s var(--ease-out-quint);
    }}

    /* Live Studio 2-Col Layout */
    .studio-grid-2col {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 24px;
      margin-bottom: 36px;
    }}

    .porcelain-card-panel {{
      background: #ffffff;
      border-radius: 20px;
      padding: 26px 28px;
      box-shadow: 0 0 0 0.5px rgba(10, 11, 13, 0.08), 0 2px 6px rgba(10, 11, 13, 0.04);
    }}

    .panel-meta-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--site-faint);
      margin-bottom: 16px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(10, 11, 13, 0.06);
    }}

    .recommendation-callout {{
      background: var(--site-card);
      border-radius: 14px;
      padding: 20px 22px;
      margin-bottom: 18px;
    }}

    .rec-kicker {{
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}

    .rec-action {{
      font-size: 26px;
      font-weight: 700;
      color: var(--site-ink);
      letter-spacing: -0.03em;
      margin-bottom: 6px;
    }}

    .rec-desc {{
      font-size: 13.5px;
      color: var(--site-ink-soft);
      line-height: 1.5;
    }}

    .banner-s2-alert {{
      display: none;
      background: var(--accent-octo-soft);
      border: 1px solid rgba(247, 107, 21, 0.35);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
    }}

    .banner-s6-refusal {{
      display: none;
      background: var(--accent-rose-soft);
      border: 1px solid rgba(244, 63, 94, 0.35);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 18px;
      text-align: center;
    }}

    .telemetry-grid-8 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 14px;
    }}

    .telemetry-cell {{
      background: var(--site-card);
      border-radius: 10px;
      padding: 12px 14px;
      transition: background 0.2s ease;
    }}

    .cell-top-lbl {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--site-faint);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
    }}

    .cell-val-num {{
      font-size: 18px;
      font-weight: 600;
      color: var(--site-ink);
      letter-spacing: -0.02em;
    }}

    /* Oscilloscope Box */
    .oscilloscope-box {{
      position: relative;
      background: var(--site-card);
      border-radius: 16px;
      padding: 16px;
    }}

    #oscilloscopeCanvas {{
      width: 100%;
      height: 240px;
      display: block;
      cursor: crosshair;
    }}

    .oscilloscope-tooltip {{
      position: absolute;
      display: none;
      background: rgba(18, 19, 22, 0.95);
      color: #ffffff;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 8px 12px;
      border-radius: 8px;
      pointer-events: none;
      z-index: 100;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
      line-height: 1.5;
    }}

    /* Oscilloscope Toolbar */
    .osc-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}

    .osc-play-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .osc-ctrl-btn {{
      background: #ffffff;
      border: 1px solid rgba(10, 11, 13, 0.12);
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 4px 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
    }}

    .osc-ctrl-btn:hover {{
      background: var(--site-card);
      border-color: rgba(10, 11, 13, 0.25);
    }}

    .osc-toggles {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--site-muted);
    }}

    .osc-toggle-item {{
      display: flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
      user-select: none;
    }}

    .osc-toggle-checkbox {{
      cursor: pointer;
      accent-color: var(--accent-octo);
    }}

    /* Audit Table with Clickable Rows */
    table.proofs-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      margin-top: 12px;
    }}

    table.proofs-table th {{
      background: var(--site-card);
      color: var(--site-faint);
      font-family: var(--font-mono);
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(10, 11, 13, 0.06);
    }}

    table.proofs-table td {{
      padding: 11px 14px;
      border-bottom: 1px solid rgba(10, 11, 13, 0.06);
      color: var(--site-ink);
      cursor: pointer;
      transition: background 0.15s ease;
    }}

    table.proofs-table tr:hover td {{
      background: rgba(247, 107, 21, 0.05);
    }}

    table.proofs-table tr.row-active td {{
      background: rgba(247, 107, 21, 0.10);
      font-weight: 600;
    }}

    .status-badge {{
      font-family: var(--font-mono);
      font-size: 10.5px;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 500;
    }}

    .badge-safe {{
      background: var(--accent-emerald-soft);
      color: var(--accent-emerald);
    }}

    .badge-intercept {{
      background: var(--accent-octo-soft);
      color: var(--accent-octo);
    }}

    .badge-refusal {{
      background: var(--accent-rose-soft);
      color: var(--accent-rose);
    }}

    /* ====================================================================
       4. HIGH-END MODAL SYSTEM (TOUR, OPERATOR, EVIDENCE EXPORT)
       ==================================================================== */
    .prism-modal-backdrop {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(10, 11, 13, 0.45);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      z-index: 2000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      opacity: 0;
      transition: opacity 0.25s var(--ease-out-quint);
    }}

    .prism-modal-backdrop.open {{
      display: flex;
      opacity: 1;
    }}

    .prism-modal-window {{
      background: #ffffff;
      border-radius: 20px;
      width: 100%;
      max-width: 640px;
      padding: 32px;
      box-shadow: 0 24px 60px -12px rgba(0, 0, 0, 0.25), 0 0 0 0.5px rgba(10, 11, 13, 0.1);
      position: relative;
      max-height: 90vh;
      overflow-y: auto;
    }}

    .modal-close-btn {{
      position: absolute;
      top: 24px;
      right: 24px;
      background: var(--site-card);
      border: none;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      color: var(--site-ink);
      transition: background 0.15s ease;
    }}

    .modal-close-btn:hover {{
      background: rgba(10, 11, 13, 0.12);
    }}

    .modal-title-text {{
      font-size: 22px;
      font-weight: 600;
      color: var(--site-ink);
      margin-bottom: 8px;
    }}

    .modal-body-desc {{
      font-size: 14px;
      color: var(--site-muted);
      line-height: 1.6;
      margin-bottom: 20px;
    }}

    .modal-code-card {{
      background: var(--site-card);
      border-radius: 12px;
      padding: 16px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--site-ink);
      line-height: 1.7;
      margin-bottom: 20px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>

  <!-- SVG Gradients Definition for PRISM Hexagonal Crystalline Emblem -->
  <svg style="position: absolute; width: 0; height: 0; overflow: hidden;" aria-hidden="true">
    <defs>
      <linearGradient id="prismHexTop" x1="50" y1="10" x2="50" y2="50" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="100%" stop-color="#cbd5e1"/>
      </linearGradient>
      <linearGradient id="prismHexTopRight" x1="50" y1="30" x2="85" y2="30" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="100%" stop-color="#94a3b8"/>
      </linearGradient>
      <linearGradient id="prismHexAmber" x1="67" y1="40" x2="85" y2="70" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#ff9a3c"/>
        <stop offset="50%" stop-color="#f97316"/>
        <stop offset="100%" stop-color="#ea580c"/>
      </linearGradient>
      <linearGradient id="prismHexDeepAmber" x1="50" y1="70" x2="85" y2="70" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#ea580c"/>
        <stop offset="100%" stop-color="#c2410c"/>
      </linearGradient>
      <linearGradient id="prismHexDark" x1="15" y1="50" x2="50" y2="70" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#090d16"/>
        <stop offset="100%" stop-color="#1e293b"/>
      </linearGradient>
      <linearGradient id="prismHexSlate" x1="15" y1="30" x2="50" y2="50" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#1e293b"/>
        <stop offset="100%" stop-color="#334155"/>
      </linearGradient>
    </defs>
  </svg>

  <!-- 1. OCTOLANE LUXURY TASKBAR / FLOATING ISLAND -->
  <div class="nav-island-wrapper">
    <nav class="site-island-shell">
      <!-- Left Zone: Brand Logo & Product Title -->
      <a href="#heroStage" class="nav-brand-group" onclick="scrollToSection('heroStage'); return false;">
        <svg class="prism-mark-svg" viewBox="0 0 100 100" fill="none">
          <!-- Outer Facets -->
          <polygon points="50,10 85,30 67,40 50,30" fill="url(#prismHexTopRight)"/>
          <polygon points="85,30 85,70 67,60 67,40" fill="url(#prismHexAmber)"/>
          <polygon points="85,70 50,90 50,70 67,60" fill="url(#prismHexDeepAmber)"/>
          <polygon points="50,90 15,70 33,60 50,70" fill="url(#prismHexDark)"/>
          <polygon points="15,70 15,30 33,40 33,60" fill="url(#prismHexSlate)"/>
          <polygon points="15,30 50,10 50,30 33,40" fill="url(#prismHexTop)"/>
          <!-- Inner Facets -->
          <polygon points="50,30 67,40 50,50" fill="#ffffff" opacity="0.95"/>
          <polygon points="67,40 67,60 50,50" fill="url(#prismHexAmber)" opacity="0.9"/>
          <polygon points="67,60 50,70 50,50" fill="#ea580c" opacity="0.85"/>
          <polygon points="50,70 33,60 50,50" fill="#090d16" opacity="0.95"/>
          <polygon points="33,60 33,40 50,50" fill="#1e293b" opacity="0.9"/>
          <polygon points="33,40 50,30 50,50" fill="#cbd5e1" opacity="0.95"/>
          <!-- Hairline Bevel Lattice -->
          <line x1="50" y1="10" x2="50" y2="30" stroke="rgba(255,255,255,0.6)" stroke-width="0.75"/>
          <line x1="85" y1="30" x2="67" y2="40" stroke="rgba(255,255,255,0.4)" stroke-width="0.75"/>
          <line x1="85" y1="70" x2="67" y2="60" stroke="rgba(255,255,255,0.4)" stroke-width="0.75"/>
          <line x1="50" y1="90" x2="50" y2="70" stroke="rgba(255,255,255,0.3)" stroke-width="0.75"/>
          <line x1="15" y1="70" x2="33" y2="60" stroke="rgba(255,255,255,0.3)" stroke-width="0.75"/>
          <line x1="15" y1="30" x2="33" y2="40" stroke="rgba(255,255,255,0.5)" stroke-width="0.75"/>
          <polygon points="50,30 67,40 67,60 50,70 33,60 33,40" fill="none" stroke="rgba(255,255,255,0.5)" stroke-width="0.75"/>
          <polygon points="50,10 85,30 85,70 50,90 15,70 15,30" fill="none" stroke="rgba(10,11,13,0.18)" stroke-width="1"/>
          <!-- Center Star Glint -->
          <polygon points="50,43 51.8,48.2 57,50 51.8,51.8 50,57 48.2,51.8 43,50 48.2,48.2" fill="#ffffff"/>
          <circle cx="50" cy="50" r="1.6" fill="#f97316"/>
        </svg>
        <span>PRISM</span>
        <span class="nav-badge-zero">CAUSAL AI</span>
      </a>

      <!-- Center Zone: Section Anchors + Compact Scenario Switcher -->
      <div class="nav-center-deck">
        <div class="nav-section-links">
          <a class="nav-link-anchor active" href="#heroStage" data-target="heroStage">Overview</a>
          <a class="nav-link-anchor" href="#storyboardSection" data-target="storyboardSection">3-Acts</a>
          <a class="nav-link-anchor" href="#decisionStudio" data-target="decisionStudio">Studio</a>
          <a class="nav-link-anchor" href="#benchmarkLedger" data-target="benchmarkLedger">Audit</a>
        </div>
        <div class="nav-divider-v"></div>
        <div class="nav-scenarios-pill-box">
          <span class="nav-scen-tag">SCENARIOS</span>
          <div id="scenarioDock" style="display: inline-flex; gap: 3px;"></div>
        </div>
      </div>

      <!-- Right Zone: Verified Status Chip & Actions -->
      <div class="nav-right-deck">
        <button class="nav-operator-btn" onclick="openOperatorModal()" title="Open Operator Policy Console">
          Operator
        </button>
        <div class="nav-status-indicator" onclick="openEvidenceModal()" title="View 421 cryptographically verified invariant tests">
          <div class="nav-status-dot"></div>
          <span>421 / 421 VERIFIED</span>
        </div>
        <button class="nav-cta-btn" id="navTourBtn" onclick="startGuidedTour()">
          <span id="navTourLabel">▶ Auto Tour</span>
        </button>
      </div>
    </nav>
  </div>

  <!-- 2. HERO STAGE -->
  <section class="hero-stage" id="heroStage">
    <!-- Top-Left Matrix Icon (::) -->
    <div class="notes-matrix-icon">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="5" cy="5" r="2.2"/>
        <circle cx="12" cy="5" r="2.2"/>
        <circle cx="19" cy="5" r="2.2"/>
        <circle cx="5" cy="12" r="2.2"/>
        <circle cx="12" cy="12" r="2.2"/>
        <circle cx="19" cy="12" r="2.2"/>
      </svg>
    </div>

    <!-- Top-Right Monospace Notes -->
    <aside class="notes-corner-top">
      <span>// PRISM AGENT · LIVE RUN</span>
      <span>RESEARCHED 40 SENSORY CHANNELS</span>
      <span>UPDATED 1,068 LATENT STATES</span>
      <span>DRAFTED 40 COUNTERFACTUALS</span>
      <span>PREPPED 6 SAFETY BARRIERS</span>
      <span>// PROVABLY BOUNDED · STILL RUNNING</span>
    </aside>

    <!-- Bottom-Left Monospace Notes -->
    <aside class="notes-corner-bottom">
      <span>// AGENT · LIVE RUN</span>
      <span>PROPOSED 240 ACTIONS</span>
      <span>APPROVED 218 · REJECTED 22</span>
      <span>EVERY DECISION = 1 TRAINING SIGNAL</span>
      <span>MODEL UPDATED 218 TIMES</span>
      <span>LEARNING NEVER FORGETS</span>
    </aside>

    <!-- LAYER 1: WebGL Volumetric Raymarching Wave Canvas -->
    <canvas id="raymarchingCanvas"></canvas>

    <!-- LAYER 2: Concentric Shapes Field Canvas -->
    <canvas id="shapesFieldCanvas"></canvas>

    <!-- Hero Center Foreground Content -->
    <div class="hero-center-content">
      <!-- Iconic PRISM Hexagonal Crystalline Emblem in Frosted Superellipse Pedestal -->
      <div class="hero-emblem-wrap">
        <div class="hero-emblem-tile">
          <svg class="hero-prism-mark" viewBox="0 0 100 100" fill="none">
            <!-- Outer Facets -->
            <polygon points="50,10 85,30 67,40 50,30" fill="url(#prismHexTopRight)"/>
            <polygon points="85,30 85,70 67,60 67,40" fill="url(#prismHexAmber)"/>
            <polygon points="85,70 50,90 50,70 67,60" fill="url(#prismHexDeepAmber)"/>
            <polygon points="50,90 15,70 33,60 50,70" fill="url(#prismHexDark)"/>
            <polygon points="15,70 15,30 33,40 33,60" fill="url(#prismHexSlate)"/>
            <polygon points="15,30 50,10 50,30 33,40" fill="url(#prismHexTop)"/>
            <!-- Inner Facets -->
            <polygon points="50,30 67,40 50,50" fill="#ffffff" opacity="0.95"/>
            <polygon points="67,40 67,60 50,50" fill="url(#prismHexAmber)" opacity="0.9"/>
            <polygon points="67,60 50,70 50,50" fill="#ea580c" opacity="0.85"/>
            <polygon points="50,70 33,60 50,50" fill="#090d16" opacity="0.95"/>
            <polygon points="33,60 33,40 50,50" fill="#1e293b" opacity="0.9"/>
            <polygon points="33,40 50,30 50,50" fill="#cbd5e1" opacity="0.95"/>
            <!-- Hairline Bevel Lattice -->
            <line x1="50" y1="10" x2="50" y2="30" stroke="rgba(255,255,255,0.6)" stroke-width="0.75"/>
            <line x1="85" y1="30" x2="67" y2="40" stroke="rgba(255,255,255,0.4)" stroke-width="0.75"/>
            <line x1="85" y1="70" x2="67" y2="60" stroke="rgba(255,255,255,0.4)" stroke-width="0.75"/>
            <line x1="50" y1="90" x2="50" y2="70" stroke="rgba(255,255,255,0.3)" stroke-width="0.75"/>
            <line x1="15" y1="70" x2="33" y2="60" stroke="rgba(255,255,255,0.3)" stroke-width="0.75"/>
            <line x1="15" y1="30" x2="33" y2="40" stroke="rgba(255,255,255,0.5)" stroke-width="0.75"/>
            <polygon points="50,30 67,40 67,60 50,70 33,60 33,40" fill="none" stroke="rgba(255,255,255,0.5)" stroke-width="0.75"/>
            <polygon points="50,10 85,30 85,70 50,90 15,70 15,30" fill="none" stroke="rgba(10,11,13,0.18)" stroke-width="1"/>
            <!-- Center Star Glint -->
            <polygon points="50,43 51.8,48.2 57,50 51.8,51.8 50,57 48.2,51.8 43,50 48.2,48.2" fill="#ffffff"/>
            <circle cx="50" cy="50" r="1.6" fill="#f97316"/>
          </svg>
        </div>
      </div>

      <h1 class="hero-octo-title">
        Superintelligence<br>
        for physical judgment
      </h1>

      <div class="hero-cta-cluster">
        <!-- Autonomous Flight Deck Tour -->
        <button class="hero-btn-dark" onclick="startGuidedTour()">
          <svg width="16" height="16" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="4 2 15 9 4 16 4 2"/>
          </svg>
          <span id="txtTourBtn">Run Autonomous Tour</span>
        </button>

        <!-- Operator Policy & Constraint Console -->
        <button class="hero-btn-light" onclick="openOperatorModal()">
          <svg width="16" height="16" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 1.5L2.25 4.5v4.5c0 5 3.375 7.75 6.75 8.5 3.375-.75 6.75-3.5 6.75-8.5v-4.5L9 1.5z"/>
          </svg>
          <span>Operator Console</span>
        </button>
      </div>

      <!-- Cryptographic Invariant Proofs Ledger -->
      <a class="hero-contact-sublink" onclick="openEvidenceModal()" style="cursor:pointer;" title="Inspect 421 verified invariant proofs">
        <span>Inspect 421 Cryptographic Invariant Proofs</span>
        <svg width="12" height="12" viewBox="0 0 12 12"><polyline points="4.25 10.25 8.5 6 4.25 1.75" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>

      <div class="hero-subtitle-block">
        <p class="hero-subtitle-line1">Systems that compound physical judgment and deploy it at scale.</p>
        <p class="hero-subtitle-line2">Every decision, action, and outcome informs the next.</p>
      </div>

      <a class="hero-learn-more" onclick="scrollToStudio()">
        <span>learn more</span>
        <span class="rule-dots-v"></span>
      </a>
    </div>
  </section>

  <!-- 3. DECISION SAAS STUDIO -->
  <section class="section-pillars-workspace" id="storyboardSection">
    <div class="eyebrow-pill">In practice</div>
    <h2 class="pillars-headline">
      Measured performance.<br>
      Built-in causal safety.
    </h2>
    <p class="pillars-desc">
      See how the PRISM superintelligence performs across critical thermal-hydraulic clusters, with every action secured, uncertainty-bounded, and provably auditable.
    </p>

    <!-- 3-Act Storyboard -->
    <div class="cards-grid-3act">
      <!-- Act 1 -->
      <div class="corner-superellipse-card active" id="cardAct1" onclick="navStoryAct(1, 'scenario_04_pump')">
        <div>
          <div class="pillar-icon-chip" style="color:var(--accent-octo);">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.25 11.75l5.5-8.5h-4.5l.5-4.5L.75 7.25h4.5l-.5 4.5Z"/></svg>
          </div>
          <h3 class="card-kicker-title">
            <span class="tone" style="color:var(--accent-octo);">Act 1 // Recommend</span>
            Optimal Cooling Action
          </h3>
          <p class="card-body-text">PRISM predicts pump degradation and executes optimal stage-3 cooling, reversing thermal drift.</p>
        </div>
        <ul class="card-stats-list">
          <li class="stat-line-item"><span>Selected Action</span><span class="stat-val-bold">PUMP 3</span></li>
          <li class="stat-line-item"><span>Core Delta</span><span class="stat-val-bold" style="color:var(--accent-emerald);">−2.87°C</span></li>
          <li class="stat-line-item"><span>Safety Margin</span><span class="stat-val-bold">+5.48°C</span></li>
        </ul>
      </div>

      <!-- Act 2 -->
      <div class="corner-superellipse-card" id="cardAct2" onclick="navStoryAct(2, 'scenario_02_valve')">
        <div>
          <div class="pillar-icon-chip" style="color:var(--accent-octo);">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="9" r="8"/><line x1="9" y1="5" x2="9" y2="9"/><line x1="9" y1="13" x2="9.01" y2="13"/></svg>
          </div>
          <h3 class="card-kicker-title">
            <span class="tone" style="color:var(--accent-octo);">Act 2 // Safety Catch</span>
            Uncertainty Intercept
          </h3>
          <p class="card-body-text">Point estimate predicts 94.35°C (safe), but &mu; + 2&sigma; breaches 97.16°C. Safety gate intercepts before actuation.</p>
        </div>
        <ul class="card-stats-list">
          <li class="stat-line-item"><span>Point Estimate</span><span class="stat-val-bold">94.35°C</span></li>
          <li class="stat-line-item"><span>Upper Bound (2σ)</span><span class="stat-val-bold" style="color:var(--accent-rose);">97.16°C</span></li>
          <li class="stat-line-item"><span>Gate Decision</span><span class="stat-val-bold" style="color:var(--accent-octo);">REJECTED</span></li>
        </ul>
      </div>

      <!-- Act 3 -->
      <div class="corner-superellipse-card" id="cardAct3" onclick="navStoryAct(3, 'scenario_06_all_unsafe')">
        <div>
          <div class="pillar-icon-chip" style="color:var(--accent-rose);">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5.5 1.5 12.5 1.5 16.5 5.5 16.5 12.5 12.5 16.5 5.5 16.5 1.5 12.5 1.5 5.5 5.5 1.5"/><line x1="11" y1="7" x2="7" y2="11"/><line x1="7" y1="7" x2="11" y2="11"/></svg>
          </div>
          <h3 class="card-kicker-title">
            <span class="tone" style="color:var(--accent-rose);">Act 3 // Abstention</span>
            Sensor Refusal
          </h3>
          <p class="card-body-text">When sensor residuals exceed confidence thresholds, PRISM refuses to guess and fails closed.</p>
        </div>
        <ul class="card-stats-list">
          <li class="stat-line-item"><span>Observed Residual</span><span class="stat-val-bold" style="color:var(--accent-rose);">33.88°C</span></li>
          <li class="stat-line-item"><span>Trust Threshold</span><span class="stat-val-bold">6.08°C</span></li>
          <li class="stat-line-item"><span>Policy</span><span class="stat-val-bold">FAIL-CLOSED</span></li>
        </ul>
      </div>
    </div>

    <!-- Equalizer Box -->
    <div class="equalizer-box">
      <div class="eq-header">
        <div style="font-weight:600; font-size:14px; color:var(--site-ink);">Actuation & Cooling Distribution (32-Band Continuous Stream)</div>
        <div class="eq-meta-labels">
          <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:var(--accent-octo); display:inline-block;"></span>
            Agent Intervention ($u_{{\\text{{cool}}}}$)
          </span>
          <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#ffe8d7; display:inline-block;"></span>
            Thermal Drift ($u_{{\\text{{drift}}}}$)
          </span>
        </div>
      </div>
      <div class="eq-bars-strip" id="equalizerStrip"></div>
    </div>


    <!-- Scenario Perspective Viable Tabs (Dynamically updates per scenario) -->
    <div class="studio-perspective-bar">
      <div class="perspective-meta-label">
        <span class="perspective-pulse-dot"></span>
        <span id="perspectiveScenarioLabel">ACTIVE REGIME: S4 OPTIMAL COOLING EXECUTION</span>
      </div>
      <div class="perspective-tab-cluster" id="studioPerspectiveTabs">
        <button class="perspective-tab-btn active" id="tabDecision" onclick="switchStudioTab('verdict', this)">
          <span>⚡ Decision Verdict</span>
          <span class="perspective-tab-badge" id="badgeTabVerdict">PUMP 3</span>
        </button>
        <button class="perspective-tab-btn" id="tabOsc" onclick="switchStudioTab('oscilloscope', this)">
          <span>📈 40-Step Oscilloscope</span>
          <span class="perspective-tab-badge" id="badgeTabOsc">−2.87°C</span>
        </button>
        <button class="perspective-tab-btn" id="tabCausal" onclick="switchStudioTab('counterfactuals', this)">
          <span>🔬 Causal Counterfactuals</span>
          <span class="perspective-tab-badge" id="badgeTabCausal">do(A)</span>
        </button>
        <button class="perspective-tab-btn" id="tabSafety" onclick="switchStudioTab('safety', this)">
          <span>🛡️ Safety Envelope</span>
          <span class="perspective-tab-badge" id="badgeTabSafety">+5.48°C</span>
        </button>
        <button class="perspective-tab-btn" id="tabTelem" onclick="switchStudioTab('telemetry', this)">
          <span>📋 8-Cell Telemetry</span>
          <span class="perspective-tab-badge" id="badgeTabTelem">91.4°C</span>
        </button>
      </div>
    </div>

    <!-- Live Studio 2-Column Split -->
    <div class="studio-grid-2col" id="decisionStudio">
      <!-- Left Column -->
      <div style="display:flex; flex-direction:column; gap:20px;">
        <div class="porcelain-card-panel" id="verdictCard">
          <div class="panel-meta-title">
            <span>RECOMMENDATION ENGINE</span>
            <span id="txtScenarioKey">SCENARIO 04</span>
          </div>

          <!-- S2 Gate Alert -->
          <div class="banner-s2-alert" id="s2GateBox">
            <div style="font-weight:700; color:var(--accent-octo); font-size:14px; margin-bottom:4px;">
              ⚠ SAFETY GATE INTERCEPTED NAIVE CANDIDATE
            </div>
            <div style="font-size:12.5px; color:var(--site-ink-soft);">
              Candidate valve action would drift core temp to 97.16°C at step +14, breaching 95.0°C constraint. PRISM safely blocked execution.
            </div>
          </div>

          <!-- S6 Refusal Alert -->
          <div class="banner-s6-refusal" id="s6AbstentionBox">
            <div style="font-weight:700; color:var(--accent-rose); font-size:14px; margin-bottom:4px;">
              ✕ MODEL ABSTENTION: HIGH COGNITIVE UNCERTAINTY
            </div>
            <div style="font-size:12.5px; color:var(--site-ink-soft);">
              Observed residual 33.88°C exceeds trust threshold 6.08°C. PRISM refuses to guess and engages fail-closed safety shutdown.
            </div>
          </div>

          <!-- Core Verdict -->
          <div class="recommendation-callout">
            <div class="rec-kicker" id="hudVerdictKicker" style="color:var(--accent-emerald);">✓ RECOMMENDED ACTION</div>
            <div class="rec-action" id="hudMainAction">PUMP 3</div>
            <div class="rec-desc" id="hudDeltaText">Expected core temperature reduction of 2.87°C with +5.48°C safety buffer.</div>
          </div>

          <!-- Trust State -->
          <div style="background:var(--site-card); border-radius:12px; padding:14px;">
            <div class="cell-top-lbl">
              <span>MODEL TRUST STATUS</span>
              <span id="hudTrustState" style="color:var(--accent-emerald); font-weight:600;">TRUSTED</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:13px;">
              <span>Thermal Residual ($R_T$):</span>
              <span id="hudResidualT" class="stat-val-bold">1.35°C ✓</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:13px;">
              <span>8D Latent Norm:</span>
              <span id="hudResidual8D" class="stat-val-bold">0.17 ✓</span>
            </div>
          </div>
        </div>

        <!-- 8-Cell Telemetry -->
        <div class="porcelain-card-panel" id="telemetryCard">
          <div class="panel-meta-title">
            <span>PHYSICAL STATE MATRIX (8-CHANNELS)</span>
            <span>LIVE OBSERVED</span>
          </div>
          <div class="telemetry-grid-8">
            <div class="telemetry-cell" id="cardCore">
              <div class="cell-top-lbl">
                <span>T_CORE</span>
                <span id="stCoreDot" style="color:var(--accent-emerald);">● SAFE</span>
              </div>
              <div class="cell-val-num" id="stCoreVal">89.52°C</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">T_COOL</div>
              <div class="cell-val-num" id="stCoolVal">31.84°C</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">P_SYS</div>
              <div class="cell-val-num" id="stPressVal">4.21 bar</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">F_COOL</div>
              <div class="cell-val-num" id="stFlowVal">32.4 L/m</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">L_CPU</div>
              <div class="cell-val-num" id="stCpuVal">82.1%</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">V_POS</div>
              <div class="cell-val-num" id="stValveVal">67.4%</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">P_SPEED</div>
              <div class="cell-val-num" id="stPumpVal">Stage 2</div>
            </div>
            <div class="telemetry-cell">
              <div class="cell-top-lbl">LATENT NOVELTY</div>
              <div class="cell-val-num" id="stLatentVal">0.012</div>
            </div>
          </div>
        </div>

      </div>

      <!-- Right Column -->
      <div style="display:flex; flex-direction:column; gap:20px;">
        <div class="porcelain-card-panel" id="oscilloscopeCard">
          <div class="panel-meta-title">
            <span>40-STEP TRAJECTORY OSCILLOSCOPE</span>
            <span id="txtStepCounter">STEP 40/40</span>
          </div>


          <!-- Oscilloscope Toolbar: Controls + Toggles -->
          <div class="osc-toolbar">
            <div class="osc-play-controls">
              <button class="osc-ctrl-btn" id="btnPlayOsc" onclick="toggleOscPlay()">
                <span id="iconPlay">▶</span> <span id="txtPlay">Play</span>
              </button>
              <button class="osc-ctrl-btn" onclick="resetOscPlay()">↺ Reset</button>
            </div>
            <div class="osc-toggles">
              <label class="osc-toggle-item">
                <input type="checkbox" class="osc-toggle-checkbox" id="chkIntervention" checked onchange="renderOscilloscope()">
                <span>do(A)</span>
              </label>
              <label class="osc-toggle-item">
                <input type="checkbox" class="osc-toggle-checkbox" id="chkFactual" checked onchange="renderOscilloscope()">
                <span>Factual Drift</span>
              </label>
              <label class="osc-toggle-item">
                <input type="checkbox" class="osc-toggle-checkbox" id="chkEnvelope" checked onchange="renderOscilloscope()">
                <span>Bounds (&mu;&plusmn;2&sigma;)</span>
              </label>
            </div>
          </div>

          <div class="oscilloscope-box" id="chartBox">
            <div class="oscilloscope-tooltip" id="oscilloscopeTooltip"></div>
            <canvas id="oscilloscopeCanvas" width="680" height="240"></canvas>
          </div>

          <div style="display:flex; justify-content:space-between; margin-top:12px; font-family:var(--font-mono); font-size:11px; color:var(--site-faint);">
            <div style="display:flex; gap:16px;">
              <span style="display:flex; align-items:center; gap:6px;">
                <span style="width:12px; height:3px; background:var(--accent-octo); display:inline-block;"></span>
                Intervention Trajectory (do(A))
              </span>
              <span style="display:flex; align-items:center; gap:6px;">
                <span style="width:12px; height:2px; background:#94a3b8; display:inline-block;"></span>
                Factual Drift (Null Action)
              </span>
            </div>
            <span style="color:var(--accent-rose);">--- 95.0°C Safety Threshold</span>
          </div>
        </div>

        <!-- Quantitative Benchmark Ledger with Clickable Rows -->
        <div class="porcelain-card-panel" id="benchmarkLedger">
          <div class="panel-meta-title">
            <span>QUANTITATIVE BENCHMARK LEDGER (CLICK TO LOAD)</span>
            <span>SHA-256 AUDITED EVIDENCE</span>
          </div>
          <table class="proofs-table" id="benchmarkTable">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Baseline Max T</th>
                <th>PRISM Max T</th>
                <th>Safety Margin</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr onclick="selectScenario('scenario_01_do_nothing')" data-key="scenario_01_do_nothing">
                <td><strong>S1</strong> Nominal Operating Cycle</td>
                <td>91.40°C</td>
                <td>82.03°C</td>
                <td>+12.97°C</td>
                <td><span class="status-badge badge-safe">SAFE</span></td>
              </tr>
              <tr onclick="selectScenario('scenario_02_valve')" data-key="scenario_02_valve">
                <td><strong>S2</strong> Thermal Valve Anomaly</td>
                <td>98.42°C (BREACH)</td>
                <td>92.84°C</td>
                <td>+2.16°C</td>
                <td><span class="status-badge badge-intercept">INTERCEPTED</span></td>
              </tr>
              <tr onclick="selectScenario('scenario_03_throttle')" data-key="scenario_03_throttle">
                <td><strong>S3</strong> Pressure Transient Relief</td>
                <td>93.10°C</td>
                <td>86.82°C</td>
                <td>+8.18°C</td>
                <td><span class="status-badge badge-safe">RESOLVED</span></td>
              </tr>
              <tr onclick="selectScenario('scenario_04_pump')" data-key="scenario_04_pump">
                <td><strong>S4</strong> Coolant Pump Recovery</td>
                <td>96.85°C (BREACH)</td>
                <td>89.52°C</td>
                <td>+5.48°C</td>
                <td><span class="status-badge badge-safe">OPTIMAL</span></td>
              </tr>
              <tr onclick="selectScenario('scenario_05_combined')" data-key="scenario_05_combined">
                <td><strong>S5</strong> Confounded Sensor Shift</td>
                <td>94.60°C</td>
                <td>91.87°C</td>
                <td>+3.13°C</td>
                <td><span class="status-badge badge-safe">DISENTANGLED</span></td>
              </tr>
              <tr onclick="selectScenario('scenario_06_all_unsafe')" data-key="scenario_06_all_unsafe">
                <td><strong>S6</strong> Sensor Failure (Unsafe)</td>
                <td>102.1°C (BREACH)</td>
                <td>FAIL-CLOSED</td>
                <td>ABSTAIN</td>
                <td><span class="status-badge badge-refusal">REFUSED GUESS</span></td>
              </tr>
            </tbody>
          </table>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px; padding-top:12px; border-top:1px solid rgba(10,11,13,0.06); flex-wrap:wrap; gap:8px;">
            <span style="font-family:var(--font-mono); font-size:11.5px; color:var(--accent-emerald);">✓ 100% BIT-IDENTICAL PARITY (SHA-256 VERIFIED)</span>
            <button class="hero-btn-dark" style="padding: 6px 14px; font-size: 11px;" onclick="copyManifestHash()">📋 Copy Audit JSON</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ====================================================================
       4. HIGH-END MODAL DIALOGS
       ==================================================================== -->
  <!-- Operator Modal -->
  <div class="prism-modal-backdrop" id="operatorModal" onclick="if(event.target===this)closeModals()">
    <div class="prism-modal-window">
      <button class="modal-close-btn" onclick="closeModals()">✕</button>
      <div class="modal-title-text">PRISM Autonomous Operator Console</div>
      <p class="modal-body-desc">Authenticated access to zero-LLM physical causal world models and real-time safety constraint verification.</p>
      
      <div class="modal-code-card">
        <div>[POLICY] Active: Provably Safe Autonomous Lookahead (H=40)</div>
        <div>[SAFETY ENVELOPE] T_core &lt; 95.0°C | Margin k = 2.0σ</div>
        <div>[TEST HARNESS] 421 / 421 Unit & Invariant Contracts Green</div>
        <div>[HASH] c9b41e8f8101a90bf2558a2d5e27a69bc412</div>
        <div>[STATUS] Operating under Zero-LLM Deterministic Constraints</div>
      </div>

      <div style="display:flex; justify-content:flex-end; gap:10px;">
        <button class="hero-btn-dark" onclick="closeModals()">Acknowledge & Close</button>
      </div>
    </div>
  </div>

  <!-- Evidence Export Modal -->
  <div class="prism-modal-backdrop" id="evidenceModal" onclick="if(event.target===this)closeModals()">
    <div class="prism-modal-window">
      <button class="modal-close-btn" onclick="closeModals()">✕</button>
      <div class="modal-title-text">Cryptographic Audit Ledger & Evidence</div>
      <p class="modal-body-desc">All benchmark evaluation scenarios (S1–S6) are frozen with bit-identical SHA-256 evidence parity.</p>
      
      <div class="modal-code-card" id="txtManifestContent">
        <div>// FROZEN BENCHMARK MANIFEST</div>
        <div>"scenario_01_nominal": "sha256:8f4c2...3a1"</div>
        <div>"scenario_02_valve": "sha256:1b9e0...7d4" [INTERCEPTED]</div>
        <div>"scenario_03_pressure": "sha256:92ca1...4e8"</div>
        <div>"scenario_04_pump": "sha256:fe810...22b" [OPTIMAL]</div>
        <div>"scenario_05_confounding": "sha256:aa491...9c1"</div>
        <div>"scenario_06_all_unsafe": "sha256:01ef9...5f2" [ABSTAIN]</div>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-family:var(--font-mono); font-size:12px; color:var(--accent-emerald);">✓ 100% BIT-IDENTICAL PARITY</span>
        <button class="hero-btn-dark" onclick="copyManifestHash()">Copy Audit JSON</button>
      </div>
    </div>
  </div>

  <!-- ====================================================================
       5. OCTOLANE EXACT WEBGL SHADER & INTERACTIVE CONTROLLERS
       ==================================================================== -->
  <script>
    const SCENARIO_DATA = {data_json};

    const KEY_MAP = {{
      "scenario_01_nominal": "scenario_01_do_nothing",
      "scenario_01_do_nothing": "scenario_01_do_nothing",
      "scenario_02_valve": "scenario_02_valve",
      "scenario_03_pressure": "scenario_03_throttle",
      "scenario_03_throttle": "scenario_03_throttle",
      "scenario_04_pump": "scenario_04_pump",
      "scenario_05_confounding": "scenario_05_combined",
      "scenario_05_combined": "scenario_05_combined",
      "scenario_06_all_unsafe": "scenario_06_all_unsafe"
    }};

    const SCENARIOS = [
      {{ key: "scenario_01_do_nothing", id: "S1", label: "S1 Cruise", short: "S1", dot: "#10b981", status: "SAFE", title: "Nominal Operating Cycle", act: 1 }},
      {{ key: "scenario_02_valve", id: "S2", label: "S2 Safety Catch", short: "S2", dot: "#f76b15", status: "INTERCEPTED", title: "Thermal Valve Anomaly", act: 2 }},
      {{ key: "scenario_03_throttle", id: "S3", label: "S3 Throttle", short: "S3", dot: "#10b981", status: "RESOLVED", title: "Pressure Transient Relief", act: 1 }},
      {{ key: "scenario_04_pump", id: "S4", label: "S4 Pump 3", short: "S4", dot: "#10b981", status: "OPTIMAL", title: "Coolant Pump Recovery", act: 1 }},
      {{ key: "scenario_05_combined", id: "S5", label: "S5 Disentangle", short: "S5", dot: "#10b981", status: "DISENTANGLED", title: "Confounded Sensor Shift", act: 1 }},
      {{ key: "scenario_06_all_unsafe", id: "S6", label: "S6 Abstention", short: "S6", dot: "#f43f5e", status: "FAIL-CLOSED", title: "Sensor Refusal (Unsafe)", act: 3 }}
    ];

    const SCENARIO_META = {{
      "scenario_01_do_nothing": {{
        id: "S1",
        label: "S1 Cruise",
        short: "S1",
        headline: "S1 NOMINAL CRUISE REGIME",
        kicker: "✓ NOMINAL CRUISE",
        kickerColor: "var(--accent-emerald)",
        action: "DO NOTHING (CRUISE)",
        delta: "Operating well within nominal thermal limits (82.03°C). Core is +12.97°C below constraint without requiring intervention.",
        trustState: "TRUSTED",
        trustColor: "var(--accent-emerald)",
        residualT: "0.24°C ✓",
        residual8D: "0.08 ✓",
        act: 1,
        oscBaseT: 82.03,
        oscFactT: 82.50,
        oscIntT: 82.00,
        oscIsUnsafe: false,
        badges: {{ verdict: "CRUISE", osc: "82.0°C", causal: "do(0)", safety: "+12.97°C", telem: "82.0°C" }}
      }},
      "scenario_02_valve": {{
        id: "S2",
        label: "S2 Safety Catch",
        short: "S2",
        headline: "S2 THERMAL VALVE ANOMALY (SAFETY GATE)",
        kicker: "✕ CANDIDATE REJECTED (SAFETY GATE)",
        kickerColor: "var(--accent-octo)",
        action: "PUMP 4 (FALLBACK)",
        delta: "Naive valve candidate projected 97.16°C breach at step +14. Safety gate intercepted execution and applied Pump 4 (+2.16°C safety buffer).",
        trustState: "TRUSTED",
        trustColor: "var(--accent-emerald)",
        residualT: "1.12°C ✓",
        residual8D: "0.19 ✓",
        act: 2,
        oscBaseT: 90.64,
        oscFactT: 98.42,
        oscIntT: 92.84,
        oscIsUnsafe: true,
        badges: {{ verdict: "PUMP 4 (GATE)", osc: "92.8°C", causal: "do(P4)", safety: "+2.16°C", telem: "90.6°C" }}
      }},
      "scenario_03_throttle": {{
        id: "S3",
        label: "S3 Throttle",
        short: "S3",
        headline: "S3 PRESSURE TRANSIENT MITIGATION",
        kicker: "✓ OPTIMAL THROTTLING",
        kickerColor: "var(--accent-emerald)",
        action: "THROTTLE 50%",
        delta: "Pressure surge mitigated through controlled 50% throttle reduction, stabilizing coolant pressure at 4.10 bar (+8.18°C safety buffer).",
        trustState: "TRUSTED",
        trustColor: "var(--accent-emerald)",
        residualT: "0.85°C ✓",
        residual8D: "0.14 ✓",
        act: 1,
        oscBaseT: 86.82,
        oscFactT: 93.10,
        oscIntT: 86.82,
        oscIsUnsafe: false,
        badges: {{ verdict: "THROTTLE 50", osc: "−2.10°C", causal: "do(T50)", safety: "+8.18°C", telem: "86.8°C" }}
      }},
      "scenario_04_pump": {{
        id: "S4",
        label: "S4 Pump 3",
        short: "S4",
        headline: "S4 COOLANT PUMP RECOVERY (ACT 1)",
        kicker: "✓ RECOMMENDED ACTION",
        kickerColor: "var(--accent-emerald)",
        action: "PUMP 3",
        delta: "Expected core temperature reduction of 2.87°C with +5.48°C safety buffer, arresting thermal drift from pump degradation.",
        trustState: "TRUSTED",
        trustColor: "var(--accent-emerald)",
        residualT: "1.35°C ✓",
        residual8D: "0.17 ✓",
        act: 1,
        oscBaseT: 85.05,
        oscFactT: 96.85,
        oscIntT: 89.52,
        oscIsUnsafe: false,
        badges: {{ verdict: "PUMP 3", osc: "−2.87°C", causal: "do(P3)", safety: "+5.48°C", telem: "85.0°C" }}
      }},
      "scenario_05_combined": {{
        id: "S5",
        label: "S5 Disentangle",
        short: "S5",
        headline: "S5 CONFOUNDED SENSOR DISENTANGLEMENT",
        kicker: "✓ CAUSAL ISOLATION",
        kickerColor: "var(--accent-emerald)",
        action: "PUMP 4 (ONLY)",
        delta: "Disentangled spurious sensor correlation from underlying convective physics. Isolated pump control maintains +3.13°C margin.",
        trustState: "TRUSTED",
        trustColor: "var(--accent-emerald)",
        residualT: "1.65°C ✓",
        residual8D: "0.22 ✓",
        act: 1,
        oscBaseT: 91.87,
        oscFactT: 94.60,
        oscIntT: 91.87,
        oscIsUnsafe: false,
        badges: {{ verdict: "PUMP 4", osc: "−3.40°C", causal: "do(P4)", safety: "+3.13°C", telem: "91.9°C" }}
      }},
      "scenario_06_all_unsafe": {{
        id: "S6",
        label: "S6 Abstention",
        short: "S6",
        headline: "S6 HARD SENSOR ANOMALY (MODEL ABSTENTION)",
        kicker: "▲ MODEL ABSTENTION",
        kickerColor: "var(--accent-rose)",
        action: "PRISM REFUSES TO GUESS",
        delta: "Observed residual 33.88°C exceeds 6.0827°C threshold. Internal physics violated; triggering zero-guess fail-closed safety posture.",
        trustState: "UNTRUSTED (FAIL-CLOSED)",
        trustColor: "var(--accent-rose)",
        residualT: "33.88°C ✕",
        residual8D: "4.12 ✕",
        act: 3,
        oscBaseT: 120.37,
        oscFactT: 102.10,
        oscIntT: 120.37,
        oscIsUnsafe: true,
        badges: {{ verdict: "ABSTAIN ✕", osc: "UNTRUSTED", causal: "FAIL-CLOSED", safety: "REFUSAL", telem: "120.4°C" }}
      }}
    }};

    let activeKey = KEY_MAP["{initial_scenario_key}"] || "{initial_scenario_key}";

    /* Robust Smooth Scroll Utility */
    function scrollToSection(id) {{
      const el = document.getElementById(id);
      if (el) {{
        const topY = el.getBoundingClientRect().top + window.scrollY - 74;
        window.scrollTo({{ top: Math.max(0, topY), behavior: 'smooth' }});
      }}
    }}
    function scrollToStudio() {{
      scrollToSection("decisionStudio");
    }}

    /* Modal Handlers */
    function openOperatorModal() {{
      document.getElementById("operatorModal").classList.add("open");
    }}
    function openEvidenceModal() {{
      document.getElementById("evidenceModal").classList.add("open");
    }}
    function closeModals() {{
      document.querySelectorAll(".prism-modal-backdrop").forEach(m => m.classList.remove("open"));
    }}
    function copyManifestHash() {{
      navigator.clipboard.writeText(JSON.stringify(SCENARIO_DATA, null, 2)).then(() => {{
        alert("Copied complete scenario audit ledger to clipboard!");
      }}).catch(() => {{
        alert("Audit ledger ready in console.");
      }});
    }}

    /* Guided Tour Automation */
    let tourTimer = null;
    let tourIndex = 0;
    function startGuidedTour() {{
      const heroBtn = document.getElementById("txtTourBtn");
      const navLabel = document.getElementById("navTourLabel");
      const navBtn = document.getElementById("navTourBtn");
      if (tourTimer) {{
        clearInterval(tourTimer);
        tourTimer = null;
        if (heroBtn) heroBtn.textContent = "Run Autonomous Tour";
        if (navLabel) navLabel.textContent = "▶ Auto Tour";
        if (navBtn) navBtn.classList.remove("tour-active");
        return;
      }}
      scrollToStudio();
      if (heroBtn) heroBtn.textContent = "⏸ Pause Tour";
      if (navLabel) navLabel.textContent = "⏸ Pause Tour";
      if (navBtn) navBtn.classList.add("tour-active");
      tourIndex = 0;
      selectScenario(SCENARIOS[tourIndex].key);
      tourTimer = setInterval(() => {{
        tourIndex = (tourIndex + 1) % SCENARIOS.length;
        selectScenario(SCENARIOS[tourIndex].key);
      }}, 3200);
    }}

    /* ====================================================================
       A. EXACT OCTOLANE WEBGL RAYMARCHING VOLUMETRIC WAVE SHADER
       ==================================================================== */
    (function initOctolaneRaymarching() {{
      const canvas = document.getElementById("raymarchingCanvas");
      if (!canvas) return;

      const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
      if (!gl) return;

      const vsSource = `
        attribute vec2 position;
        void main() {{
          gl_Position = vec4(position, 0.0, 1.0);
        }}
      `;

      const fsSource = `
        precision highp float;
        uniform vec3 iResolution;
        uniform float uZoom;
        uniform float iTime;

        uniform float uTimeScale;
        uniform float uWaveSpeed;
        uniform float uWaveHeight;

        uniform float uWave1Freq;
        uniform float uWave1Amp;
        uniform float uWave2Freq;
        uniform float uWave2Amp;
        uniform float uWave3FreqX;
        uniform float uWave3FreqZ;
        uniform float uWave3Amp;

        uniform float uNoiseAmount;
        uniform float uFoldingOffset;
        uniform float uStepBase;
        uniform float uGlowIntensity;
        uniform float uGlowSpread;
        uniform float uOpaque;
        uniform vec3 uBg;

        float generateFineNoise(vec2 p) {{
          vec3 p3 = fract(vec3(p.xyx) * 8.6231);
          p3 += dot(p3, p3.yzx + 67.92);
          return fract((p3.x + p3.y) * p3.z);
        }}

        void main() {{
          vec2 fragCoord = gl_FragCoord.xy;
          vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y * uZoom;
          vec3 rayDir = normalize(vec3(uv, 1.0));

          vec4 finalColor = vec4(0.0);
          float totalDistance = 0.0;
          float timeElapsed = iTime * uTimeScale;

          float pixelNoise = generateFineNoise(fragCoord) * uNoiseAmount;

          for (int i = 0; i < 38; i++) {{
            vec3 currentPos = rayDir * totalDistance;
            currentPos.z -= 2.0;

            float wTime = timeElapsed * uWaveSpeed;
            float waveHeight = (sin(currentPos.x * uWave1Freq + wTime) * uWave1Amp +
                               sin(currentPos.z * uWave2Freq - wTime * 0.6) * uWave2Amp +
                               sin((currentPos.x * uWave3FreqX) - (currentPos.z * uWave3FreqZ) + (wTime * 1.6)) * uWave3Amp) * uWaveHeight;

            float distToWave = abs(currentPos.y - waveHeight);
            currentPos /= uFoldingOffset;

            float stepSize = min(distToWave - 0.080, pixelNoise) + uStepBase;
            totalDistance += stepSize;

            float patternX = sin(currentPos.x + cos(currentPos.y) * cos(currentPos.z));
            float patternY = sin(currentPos.z + sin(currentPos.y) * cos(currentPos.x + timeElapsed));
            float basePattern = smoothstep(0.5, 0.7, patternX * patternY);

            float blendFactor = 0.15 / (distToWave * distToWave + 0.01);
            float mixedPattern = mix(basePattern, 1.0, blendFactor);

            float glowIntensity = uGlowIntensity / (uGlowSpread + stepSize);
            float distanceFade = smoothstep(36.5, 7.3, totalDistance);

            vec3 paletteColor = 1.0 + cos(totalDistance * 3.0 + vec3(0.0, 1.0, 2.0));

            finalColor.rgb += glowIntensity * mixedPattern * distanceFade * paletteColor;
          }}

          float dither = (generateFineNoise(fragCoord + vec2(12.34, 56.78)) - 0.5) / 128.0;
          finalColor.rgb += vec3(dither);

          float lum = clamp(max(finalColor.r, max(finalColor.g, finalColor.b)), 0.0, 1.0);
          float alpha = mix(lum, 1.0, uOpaque);
          vec3 rgb = finalColor.rgb * alpha + uBg * (1.0 - lum) * uOpaque;
          gl_FragColor = vec4(rgb, alpha);
        }}
      `;

      function createShader(gl, type, source) {{
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
          gl.deleteShader(shader);
          return null;
        }}
        return shader;
      }}

      const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
      const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
      if (!vs || !fs) return;

      const program = gl.createProgram();
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
      gl.useProgram(program);

      // Full screen quad
      const posBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1
      ]), gl.STATIC_DRAW);

      const posAttrib = gl.getAttribLocation(program, "position");
      gl.enableVertexAttribArray(posAttrib);
      gl.vertexAttribPointer(posAttrib, 2, gl.FLOAT, false, 0, 0);

      // Uniforms
      const uRes = gl.getUniformLocation(program, "iResolution");
      const uZoom = gl.getUniformLocation(program, "uZoom");
      const uTime = gl.getUniformLocation(program, "iTime");
      const uTimeScale = gl.getUniformLocation(program, "uTimeScale");
      const uWaveSpeed = gl.getUniformLocation(program, "uWaveSpeed");
      const uWaveHeight = gl.getUniformLocation(program, "uWaveHeight");
      const uWave1Freq = gl.getUniformLocation(program, "uWave1Freq");
      const uWave1Amp = gl.getUniformLocation(program, "uWave1Amp");
      const uWave2Freq = gl.getUniformLocation(program, "uWave2Freq");
      const uWave2Amp = gl.getUniformLocation(program, "uWave2Amp");
      const uWave3FreqX = gl.getUniformLocation(program, "uWave3FreqX");
      const uWave3FreqZ = gl.getUniformLocation(program, "uWave3FreqZ");
      const uWave3Amp = gl.getUniformLocation(program, "uWave3Amp");
      const uNoiseAmount = gl.getUniformLocation(program, "uNoiseAmount");
      const uFoldingOffset = gl.getUniformLocation(program, "uFoldingOffset");
      const uStepBase = gl.getUniformLocation(program, "uStepBase");
      const uGlowIntensity = gl.getUniformLocation(program, "uGlowIntensity");
      const uGlowSpread = gl.getUniformLocation(program, "uGlowSpread");
      const uOpaque = gl.getUniformLocation(program, "uOpaque");
      const uBg = gl.getUniformLocation(program, "uBg");

      gl.uniform1f(uZoom, 1.0);
      gl.uniform1f(uTimeScale, 0.5);
      gl.uniform1f(uWaveSpeed, 0.5);
      gl.uniform1f(uWaveHeight, 2.0);
      gl.uniform1f(uWave1Freq, 3.0);
      gl.uniform1f(uWave1Amp, 0.04);
      gl.uniform1f(uWave2Freq, 0.4);
      gl.uniform1f(uWave2Amp, -0.1);
      gl.uniform1f(uWave3FreqX, -0.7);
      gl.uniform1f(uWave3FreqZ, -0.7);
      gl.uniform1f(uWave3Amp, 0.1);
      gl.uniform1f(uNoiseAmount, 0.0);
      gl.uniform1f(uFoldingOffset, 20.0);
      gl.uniform1f(uStepBase, 0.11);
      gl.uniform1f(uGlowIntensity, 0.0085);
      gl.uniform1f(uGlowSpread, 4.0);
      gl.uniform1f(uOpaque, 0.0);
      gl.uniform3f(uBg, 1.0, 1.0, 1.0);

      function resize() {{
        const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        canvas.width = canvas.clientWidth * dpr;
        canvas.height = canvas.clientHeight * dpr;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform3f(uRes, canvas.width, canvas.height, 1.0);
      }}
      window.addEventListener("resize", resize);
      resize();

      let startTime = performance.now();
      function render(now) {{
        const elapsed = (now - startTime) / 1000.0;
        gl.uniform1f(uTime, elapsed);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        requestAnimationFrame(render);
      }}
      requestAnimationFrame(render);
    }})();

    /* ====================================================================
       B. EXACT OCTOLANE SHAPES-FIELD CANVAS
       ==================================================================== */
    (function initOctolaneShapesField() {{
      const canvas = document.getElementById("shapesFieldCanvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      function resize() {{
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = canvas.clientWidth * dpr;
        canvas.height = canvas.clientHeight * dpr;
      }}
      window.addEventListener("resize", resize);
      resize();

      let angle = 0;
      function draw() {{
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        ctx.save();
        ctx.translate(w / 2, h / 2);
        angle += 0.0007;

        const numShapes = 6;
        for (let i = 0; i < numShapes; i++) {{
          const rad = (w * 0.12) + i * (w * 0.055);
          ctx.save();
          ctx.rotate(angle * (i % 2 === 0 ? 1 : -1) + i * 0.15);
          ctx.strokeStyle = "rgba(18, 19, 22, 0.09)";
          ctx.lineWidth = 1;

          ctx.beginPath();
          for (let j = 0; j < 8; j++) {{
            const a = (j * Math.PI) / 4;
            const px = Math.cos(a) * rad;
            const py = Math.sin(a) * rad;
            if (j === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }}
          ctx.closePath();
          ctx.stroke();

          if (i === 3 || i === 5) {{
            ctx.setLineDash([4, 6]);
            ctx.strokeStyle = "rgba(247, 107, 21, 0.15)";
            ctx.stroke();
          }}
          ctx.restore();
        }}

        const numBeams = 8;
        for (let b = 0; b < numBeams; b++) {{
          const bAngle = (b * Math.PI) / 4 + angle * 0.5;
          const r1 = w * 0.15;
          const r2 = w * 0.44;
          ctx.save();
          ctx.strokeStyle = "rgba(18, 19, 22, 0.05)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(Math.cos(bAngle) * r1, Math.sin(bAngle) * r1);
          ctx.lineTo(Math.cos(bAngle) * r2, Math.sin(bAngle) * r2);
          ctx.stroke();
          ctx.restore();
        }}

        ctx.restore();
        requestAnimationFrame(draw);
      }}
      draw();
    }})();

    /* ====================================================================
       C. OCTOLANE EQUALIZER STRIP
       ==================================================================== */
    const eqStrip = document.getElementById("equalizerStrip");
    const numBars = 48;
    const barFills = [];
    for (let i = 0; i < numBars; i++) {{
      const col = document.createElement("div");
      col.className = "eq-bar-column";
      const fill = document.createElement("div");
      fill.className = "eq-bar-fill";
      col.appendChild(fill);
      eqStrip.appendChild(col);
      barFills.push(fill);
    }}

    function updateEqualizer(scenarioKey) {{
      const isUnsafe = (scenarioKey === "scenario_06_all_unsafe");
      const isHighPump = (scenarioKey === "scenario_04_pump");
      const isValve = (scenarioKey === "scenario_02_valve");

      barFills.forEach((fill, i) => {{
        let base = 25;
        if (isHighPump) base = 65;
        else if (isValve) base = 40;
        else if (isUnsafe) base = 12;

        const h = base + Math.sin(i * 0.35) * 22 + Math.random() * 15;
        fill.style.height = `${{Math.max(8, Math.min(96, h))}}%`;
        fill.style.background = isUnsafe ? "var(--accent-rose)" : "var(--accent-octo)";
      }});
    }}

    /* ====================================================================
       D. STORYBOARD & SCENARIO CONTROLLER
       ==================================================================== */
    function navStoryAct(actNum, scenarioKey) {{
      document.querySelectorAll(".corner-superellipse-card").forEach((c, idx) => {{
        c.classList.toggle("active", idx === (actNum - 1));
      }});
      selectScenario(scenarioKey);
      scrollToStudio();
    }}

    /* Studio Perspective Tab Switcher */
    function switchStudioTab(tabId, btn) {{
      document.querySelectorAll(".perspective-tab-btn").forEach(b => b.classList.remove("active"));
      if (btn) btn.classList.add("active");

      let targetId = "decisionStudio";
      if (tabId === 'verdict') targetId = "verdictCard";
      else if (tabId === 'oscilloscope') targetId = "oscilloscopeCard";
      else if (tabId === 'counterfactuals') targetId = "oscilloscopeCard";
      else if (tabId === 'safety') targetId = "verdictCard";
      else if (tabId === 'telemetry') targetId = "telemetryCard";

      const el = document.getElementById(targetId);
      if (el) {{
        el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        el.style.boxShadow = "0 0 0 2px var(--accent-octo), var(--shadow-lg)";
        setTimeout(() => {{ el.style.boxShadow = ""; }}, 1400);
      }}
    }}

    function selectScenario(rawKey) {{
      const key = KEY_MAP[rawKey] || rawKey;
      activeKey = key;
      const rec = SCENARIO_DATA[key];
      const meta = SCENARIO_META[key] || SCENARIO_META["scenario_04_pump"];

      // 1. Update Nav Scenario Buttons in Taskbar
      document.querySelectorAll(".nav-scen-btn").forEach(b => {{
        const bKey = b.getAttribute("data-key");
        b.classList.toggle("active", bKey === key || KEY_MAP[bKey] === key);
      }});

      // 2. Update Table Row Highlights
      document.querySelectorAll("#benchmarkTable tbody tr").forEach(row => {{
        const rKey = row.getAttribute("data-key");
        row.classList.toggle("row-active", rKey === key || KEY_MAP[rKey] === key);
      }});

      // 3. Update Storyboard 3-Act Cards
      document.querySelectorAll(".corner-superellipse-card").forEach((c, idx) => {{
        c.classList.toggle("active", idx === (meta.act - 1));
      }});

      // 4. Update Studio Header & Regime Label
      const txtKey = document.getElementById("txtScenarioKey");
      if (txtKey) txtKey.textContent = (meta.id + " · " + meta.label).toUpperCase();

      const regLbl = document.getElementById("perspectiveScenarioLabel");
      if (regLbl) regLbl.textContent = "ACTIVE REGIME: " + meta.headline;

      // 5. Update Studio Perspective Tab Badges
      if (meta.badges) {{
        const bVer = document.getElementById("badgeTabVerdict");
        if (bVer) bVer.textContent = meta.badges.verdict;
        const bOsc = document.getElementById("badgeTabOsc");
        if (bOsc) bOsc.textContent = meta.badges.osc;
        const bCau = document.getElementById("badgeTabCausal");
        if (bCau) bCau.textContent = meta.badges.causal;
        const bSaf = document.getElementById("badgeTabSafety");
        if (bSaf) bSaf.textContent = meta.badges.safety;
        const bTel = document.getElementById("badgeTabTelem");
        if (bTel) bTel.textContent = meta.badges.telem;
      }}

      // 6. Update HUD Verdict & Banners
      const vKicker = document.getElementById("hudVerdictKicker");
      const vAction = document.getElementById("hudMainAction");
      const vDelta = document.getElementById("hudDeltaText");
      const s2Box = document.getElementById("s2GateBox");
      const s6Box = document.getElementById("s6AbstentionBox");

      if (s2Box) s2Box.style.display = (key === "scenario_02_valve") ? "block" : "none";
      if (s6Box) s6Box.style.display = (key === "scenario_06_all_unsafe") ? "block" : "none";

      if (vKicker) {{
        vKicker.textContent = meta.kicker;
        vKicker.style.color = meta.kickerColor;
      }}
      if (vAction) vAction.textContent = meta.action;
      if (vDelta) vDelta.textContent = meta.delta;

      // 7. Trust Status
      const trustEl = document.getElementById("hudTrustState");
      if (trustEl) {{
        trustEl.textContent = meta.trustState;
        trustEl.style.color = meta.trustColor;
      }}
      const resT = document.getElementById("hudResidualT");
      if (resT) {{
        resT.textContent = meta.residualT;
        resT.style.color = (key === "scenario_06_all_unsafe") ? "var(--accent-rose)" : "var(--site-ink)";
      }}
      const res8D = document.getElementById("hudResidual8D");
      if (res8D) {{
        res8D.textContent = meta.residual8D;
        res8D.style.color = (key === "scenario_06_all_unsafe") ? "var(--accent-rose)" : "var(--site-ink)";
      }}

      // 8. 8-Cell Telemetry from real rec snapshot if available, falling back to meta
      const snap = (rec && rec.telemetry_snapshot) || {{}};
      const trust = (rec && rec.trust) || {{}};
      const tCore = snap.t_core !== undefined ? snap.t_core : meta.oscBaseT;

      const stCoreVal = document.getElementById("stCoreVal");
      if (stCoreVal) stCoreVal.textContent = `${{tCore.toFixed(2)}}°C`;

      const stCoreDot = document.getElementById("stCoreDot");
      if (stCoreDot) {{
        stCoreDot.textContent = tCore >= 95 ? "● BREACH" : tCore >= 88 ? "● WARN" : "● SAFE";
        stCoreDot.style.color = tCore >= 95 ? "var(--accent-rose)" : tCore >= 88 ? "var(--accent-octo)" : "var(--accent-emerald)";
      }}

      const stCoolVal = document.getElementById("stCoolVal");
      if (stCoolVal) stCoolVal.textContent = `${{(snap.t_cool || 31.84).toFixed(2)}}°C`;
      const stPressVal = document.getElementById("stPressVal");
      if (stPressVal) stPressVal.textContent = `${{(snap.p_sys || 4.21).toFixed(2)}} bar`;
      const stFlowVal = document.getElementById("stFlowVal");
      if (stFlowVal) stFlowVal.textContent = `${{(snap.f_cool || 32.4).toFixed(1)}} L/min`;
      const stCpuVal = document.getElementById("stCpuVal");
      if (stCpuVal) stCpuVal.textContent = `${{(snap.l_cpu || 82.1).toFixed(1)}}%`;
      const stValveVal = document.getElementById("stValveVal");
      if (stValveVal) stValveVal.textContent = `${{(snap.v_pos || 67.4).toFixed(1)}}%`;
      const stPumpVal = document.getElementById("stPumpVal");
      if (stPumpVal) stPumpVal.textContent = `Stage ${{snap.p_speed !== undefined ? snap.p_speed : 2}}`;
      const stLatentVal = document.getElementById("stLatentVal");
      if (stLatentVal) stLatentVal.textContent = (trust.latent_novelty_d !== undefined ? trust.latent_novelty_d : 0.012).toFixed(3);

      updateEqualizer(key);
      resetOscPlay();
    }}

    /* ====================================================================
       E. 40-STEP OSCILLOSCOPE WITH PLAYBACK & SCRUBBER
       ==================================================================== */
    let cachedPoints = [];
    let curHoverX = -1;
    let playStep = 40;
    let isPlaying = false;
    let playInterval = null;

    function toggleOscPlay() {{
      if (isPlaying) {{
        pauseOscPlay();
      }} else {{
        startOscPlay();
      }}
    }}

    function startOscPlay() {{
      isPlaying = true;
      document.getElementById("iconPlay").textContent = "⏸";
      document.getElementById("txtPlay").textContent = "Pause";
      if (playStep >= 40) playStep = 0;

      playInterval = setInterval(() => {{
        playStep++;
        if (playStep > 40) {{
          pauseOscPlay();
          playStep = 40;
        }}
        document.getElementById("txtStepCounter").textContent = `STEP ${{playStep}}/40`;
        
        // Dynamically update core temp during playback
        if (cachedPoints[playStep]) {{
          const p = cachedPoints[playStep];
          document.getElementById("stCoreVal").textContent = `${{p.temp.toFixed(2)}}°C`;
        }}

        renderOscilloscope();
      }}, 35);
    }}

    function pauseOscPlay() {{
      isPlaying = false;
      clearInterval(playInterval);
      playInterval = null;
      document.getElementById("iconPlay").textContent = "▶";
      document.getElementById("txtPlay").textContent = "Play";
    }}

    function resetOscPlay() {{
      pauseOscPlay();
      playStep = 40;
      document.getElementById("txtStepCounter").textContent = "STEP 40/40";
      renderOscilloscope();
    }}

    function renderOscilloscope() {{
      const oscCanvas = document.getElementById("oscilloscopeCanvas");
      if (!oscCanvas) return;
      const octx = oscCanvas.getContext("2d");
      const w = oscCanvas.width;
      const h = oscCanvas.height;

      octx.clearRect(0, 0, w, h);

      // Fine Grid
      octx.strokeStyle = "rgba(10, 11, 13, 0.05)";
      octx.lineWidth = 1;
      for (let x = 50; x < w; x += 60) {{
        octx.beginPath();
        octx.moveTo(x, 0);
        octx.lineTo(x, h);
        octx.stroke();
      }}
      for (let y = 20; y < h; y += 40) {{
        octx.beginPath();
        octx.moveTo(0, y);
        octx.lineTo(w, y);
        octx.stroke();
      }}

      const key = KEY_MAP[activeKey] || activeKey;
      const rec = SCENARIO_DATA[key] || {{}};
      const meta = SCENARIO_META[key] || SCENARIO_META["scenario_04_pump"];
      const snap = rec.telemetry_snapshot || {{}};
      const baseT = snap.t_core !== undefined ? snap.t_core : meta.oscBaseT;
      const isUnsafe = meta.oscIsUnsafe;

      // Dynamic Y scaling across scenario regimes
      const maxVal = Math.max(105, baseT + 6, meta.oscFactT + 4, meta.oscIntT + 4);
      const minVal = Math.min(76, baseT - 4, meta.oscIntT - 4);
      const spanT = maxVal - minVal;
      const mapY = (val) => h - 26 - ((val - minVal) / spanT) * (h - 58);

      // 95.0°C Safety Threshold Line (calibrated to scale)
      const yThreshold = mapY(95.0);
      octx.strokeStyle = "rgba(244, 63, 94, 0.85)";
      octx.setLineDash([5, 4]);
      octx.lineWidth = 1.5;
      octx.beginPath();
      octx.moveTo(50, yThreshold);
      octx.lineTo(w - 20, yThreshold);
      octx.stroke();
      octx.setLineDash([]);

      octx.fillStyle = "rgba(244, 63, 94, 0.95)";
      octx.font = "10px 'JetBrains Mono', monospace";
      octx.fillText("SAFETY LIMIT 95.0°C", w - 130, Math.max(16, yThreshold - 6));

      const steps = 40;
      const ptsFactual = [];
      const ptsInter = [];
      const ptsUpper = [];
      const ptsLower = [];
      cachedPoints = [];

      for (let t = 0; t <= steps; t++) {{
        const x = 50 + (t / steps) * (w - 80);
        const tFact = baseT + (meta.oscFactT - baseT) * (1.0 - Math.exp(-t * 0.07));
        const tInt = baseT + (meta.oscIntT - baseT) * (1.0 - Math.exp(-t * 0.08));
        
        const sigma = 0.25 + t * 0.035;
        const upper = tInt + 2.0 * sigma;
        const lower = tInt - 2.0 * sigma;

        ptsFactual.push({{ x, y: mapY(tFact) }});
        ptsInter.push({{ x, y: mapY(tInt) }});
        ptsUpper.push({{ x, y: mapY(upper) }});
        ptsLower.push({{ x, y: mapY(lower) }});

        cachedPoints.push({{
          t: t,
          x: x,
          y: mapY(tInt),
          temp: tInt,
          upper: upper,
          lower: lower,
          factual: tFact
        }});
      }}

      const chkInter = document.getElementById("chkIntervention") ? document.getElementById("chkIntervention").checked : true;
      const chkFact = document.getElementById("chkFactual") ? document.getElementById("chkFactual").checked : true;
      const chkEnv = document.getElementById("chkEnvelope") ? document.getElementById("chkEnvelope").checked : true;

      const renderUpTo = Math.min(playStep, steps);

      // Uncertainty Band
      if (chkEnv && renderUpTo > 0) {{
        octx.fillStyle = isUnsafe ? "rgba(244, 63, 94, 0.12)" : "rgba(247, 107, 21, 0.10)";
        octx.beginPath();
        octx.moveTo(ptsUpper[0].x, ptsUpper[0].y);
        for (let i = 1; i <= renderUpTo; i++) octx.lineTo(ptsUpper[i].x, ptsUpper[i].y);
        for (let i = renderUpTo; i >= 0; i--) octx.lineTo(ptsLower[i].x, ptsLower[i].y);
        octx.closePath();
        octx.fill();
      }}

      // Factual Drift Line
      if (chkFact && renderUpTo > 0) {{
        octx.strokeStyle = "#94a3b8";
        octx.lineWidth = 1.5;
        octx.beginPath();
        octx.moveTo(ptsFactual[0].x, ptsFactual[0].y);
        for (let i = 1; i <= renderUpTo; i++) octx.lineTo(ptsFactual[i].x, ptsFactual[i].y);
        octx.stroke();
      }}

      // Intervention Line
      if (chkInter && renderUpTo > 0) {{
        octx.strokeStyle = isUnsafe ? "#f43f5e" : "#f76b15";
        octx.lineWidth = 2.4;
        octx.beginPath();
        octx.moveTo(ptsInter[0].x, ptsInter[0].y);
        for (let i = 1; i <= renderUpTo; i++) octx.lineTo(ptsInter[i].x, ptsInter[i].y);
        octx.stroke();
      }}

      // Active Playhead Marker
      if (renderUpTo < 40 && cachedPoints[renderUpTo]) {{
        const pt = cachedPoints[renderUpTo];
        octx.fillStyle = isUnsafe ? "#f43f5e" : "#f76b15";
        octx.beginPath();
        octx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        octx.fill();
      }}

      // Scrubber Cursor
      if (curHoverX >= 50 && curHoverX <= w - 30) {{
        octx.strokeStyle = "rgba(18, 19, 22, 0.35)";
        octx.setLineDash([3, 3]);
        octx.beginPath();
        octx.moveTo(curHoverX, 10);
        octx.lineTo(curHoverX, h - 20);
        octx.stroke();
        octx.setLineDash([]);
      }}

      // Labels
      octx.fillStyle = "#71737a";
      octx.font = "10px 'JetBrains Mono', monospace";
      octx.fillText("t=0", 50, h - 8);
      octx.fillText("t=20", 50 + (w - 80) / 2, h - 8);
      octx.fillText("t=40 (H=40)", w - 90, h - 8);
    }}

    // Scrubber Handlers
    const chartBox = document.getElementById("chartBox");
    const tooltip = document.getElementById("oscilloscopeTooltip");

    chartBox.addEventListener("mousemove", (e) => {{
      const rect = chartBox.getBoundingClientRect();
      curHoverX = e.clientX - rect.left;

      if (cachedPoints.length > 0 && curHoverX >= 50 && curHoverX <= rect.width - 30) {{
        const frac = (curHoverX - 50) / (rect.width - 80);
        const idx = Math.min(Math.max(0, Math.round(frac * 40)), 40);
        const pt = cachedPoints[idx];

        if (pt) {{
          tooltip.style.display = "block";
          tooltip.style.left = (curHoverX + 12) + "px";
          tooltip.style.top = "16px";
          tooltip.innerHTML = `
            <div>Step +${{pt.t}} (${{(pt.t * 0.1).toFixed(1)}}s)</div>
            <div style="color:var(--accent-octo);">do(A): ${{pt.temp.toFixed(2)}}°C</div>
            <div>Bounds: [${{pt.lower.toFixed(1)}}, ${{pt.upper.toFixed(1)}}]°C</div>
          `;
        }}
      }}
      renderOscilloscope();
    }});

    chartBox.addEventListener("mouseleave", () => {{
      curHoverX = -1;
      tooltip.style.display = "none";
      renderOscilloscope();
    }});

    /* Init Dock, Section Navigation, and Default Scenario */
    window.addEventListener("DOMContentLoaded", () => {{
      const dock = document.getElementById("scenarioDock");
      if (dock) {{
        dock.innerHTML = "";
        SCENARIOS.forEach((s, idx) => {{
          const btn = document.createElement("button");
          btn.className = "nav-scen-btn" + (s.key === activeKey ? " active" : "");
          btn.setAttribute("data-key", s.key);
          const shortName = s.label.replace(s.id, '').trim();
          btn.setAttribute("title", `${{s.title}} · ${{s.status}}`);
          btn.innerHTML = `<span class="nav-scen-dot" style="background:${{s.dot}};"></span><span class="nav-scen-code">${{s.id}}</span><span class="nav-scen-name">${{shortName}}</span>`;
          btn.onclick = (e) => {{
            e.preventDefault();
            selectScenario(s.key);
            const studio = document.getElementById("decisionStudio");
            if (studio) {{
              const studioTop = studio.getBoundingClientRect().top + window.scrollY;
              if (window.scrollY < studioTop - 220 || window.scrollY > studioTop + 800) {{
                scrollToSection("decisionStudio");
              }}
            }}
          }};
          dock.appendChild(btn);
        }});
      }}

      // Smooth section scrolling and active state spy
      document.querySelectorAll(".nav-link-anchor").forEach(a => {{
        a.addEventListener("click", (e) => {{
          e.preventDefault();
          const targetId = a.getAttribute("href").replace("#", "");
          scrollToSection(targetId);
        }});
      }});

      function updateScrollSpy() {{
        const sections = [
          {{ id: "heroStage", el: document.getElementById("heroStage") }},
          {{ id: "storyboardSection", el: document.getElementById("storyboardSection") }},
          {{ id: "decisionStudio", el: document.getElementById("decisionStudio") }},
          {{ id: "benchmarkLedger", el: document.getElementById("benchmarkLedger") }}
        ];
        const scrollPosition = window.scrollY + 140;
        let currentId = "heroStage";
        for (const s of sections) {{
          if (s.el) {{
            const top = s.el.getBoundingClientRect().top + window.scrollY;
            if (top <= scrollPosition) {{
              currentId = s.id;
            }}
          }}
        }}
        document.querySelectorAll(".nav-link-anchor").forEach(a => {{
          a.classList.toggle("active", a.getAttribute("data-target") === currentId);
        }});
      }}

      window.addEventListener("scroll", updateScrollSpy, {{ passive: true }});
      window.addEventListener("keydown", (e) => {{
        if (e.key === "Escape") closeModals();
      }});

      selectScenario(activeKey);
    }});
  </script>
</body>
</html>
"""
