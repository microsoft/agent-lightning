# Agent-lightning Dashboard — Detailed Design Spec

Tech stack: **React + Mantine UI**, **Redux Toolkit**, **React Router**, **Storybook**. Backend: **FastAPI (uvicorn)**.

## App architecture (high level)

### AppShell layout

`AppShell` with persistent `Navbar` (left), optional `Header` (top; can be minimal), and main content as a `Router` outlet.

### Navbar content

* Brand row: `[ActionIcon] Agent-lightning Dashboard`
* `Navbar.Section` for main nav: `Rollouts`, `Resources`, `Traces`, `Settings` (use `NavLink`s that preserve current query when it makes sense)
* Footer: connection indicator (bound to `ServerConfig.baseUrl` & health state). When refreshing, the status will be a blinking "Refreshing".

### State/data: Redux Toolkit + RTK Query

* Slice: `serverConfig` (`baseUrl`, `autoRefreshMs`)
* UI slices: `ui/toasts`, `ui/drawers`, `ui/theme` (dark/light), `ui/routerState` (deep-link param mirrors where local state is needed)
* Maybe some more on the data model later...

### URL ↔ state sync

For the first iteration, no state will appear in the URL, other than pages like `/rollouts`, `/resources`, `/traces`, `/settings`.

Other states like table sorting status are just cached in the Redux store (or just ephemeral on the components).

### Async UX

* Every list view: `Skeleton` → Empty (illustration + refresh button) → Error (`Alert`)
* Global `showToast` via Mantine `notifications.show()` or `showNotification`

### Right drawer pattern

Single `Drawer` component that is controlled by Redux state. Content swappable by drawer type.

### Code/JSON

* `@mantine/code-highlight` for pretty JSON blocks
* `JsonInput` for editable JSON only in **Settings** (not for data returned by APIs)

## Data Model

See `src/types.ts` for the data model including `Attempt`, `Rollout`, `Resources`.

## API Contracts (FastAPI)

The following APIs are implemented in the Python backend.

### Health

* **GET `/health`** — Simple health check endpoint to confirm the server is alive.

### Queue Management

* **POST `/queues/rollouts/enqueue`** — Add a new rollout to the queue (`status="queuing"`), no attempt created yet.
* **POST `/queues/rollouts/dequeue`** — Claim the oldest queued rollout; transitions to `preparing` and creates a new attempt.

### Rollouts

* **POST `/rollouts`** — Start a new rollout immediately and create its first attempt (`status="preparing"`).
* **GET `/rollouts`** — List all rollouts. Returns a list of rollouts and their latest attempts.
* **POST `/rollouts/search`** — Search rollouts by `status` or specific rollout IDs. Expects a JSON body like `{"status": ["succeeded", "failed"]}`. Returns a list of rollouts and their latest attempts.
* **GET `/rollouts/{rollout_id}`** — Retrieve a rollout by its ID. Returns `null` if not found. Returns the rollout and its latest attempt.
* **POST `/rollouts/{rollout_id}`** — Update rollout metadata or status (can move it to terminal or queued states).

### Attempts

* **GET `/rollouts/{rollout_id}/attempts`** — List all attempts for a given rollout (ordered oldest → newest).
* **GET `/rollouts/{rollout_id}/attempts/latest`** — Get the most recent attempt for a rollout. Returns `null` if not found.
* **POST `/rollouts/{rollout_id}/attempts`** — Manually create a new retry attempt.
* **POST `/rollouts/{rollout_id}/attempts/{attempt_id}`** — Update attempt state, worker ID, heartbeat time, or metadata.

### Resources

* **POST `/resources`** — Create a new immutable resource snapshot and mark it as the latest version.
* **POST `/resources/{resources_id}`** — Update an existing resource snapshot and mark it as latest.
* **GET `/resources/{resources_id}`** — Fetch a resource snapshot by ID.
* **GET `/resources/latest`** — Retrieve the most recent (default) resource snapshot.

### Spans & Telemetry

* **POST `/spans`** — Record a telemetry span from a rollout attempt; also updates heartbeat and state.
* **GET `/spans`** — Query stored spans by rollout (and optionally attempt). Accepts GET params like `?rollout_id=...&attempt_id=...`. Currently `rollout_id` is required.
* **POST `/spans/next`** — Get the next sequential ID for span ordering.

### Wait / Synchronization

* **POST `/waits/rollouts`** — Wait until one or more rollouts finish (`succeeded`, `failed`, or `cancelled`), or timeout expires.

## Pages (Draft)

### Rollouts

Rollout is a large data table that displays all the rollouts from the backend (not only those in the queue). It supports pagination (page size = 100 by default), sorting, filtering on categorical columns and searching on Rollout ID.

By default, the rollouts and their associated latest attempts will be fetched and displayed in the table. They will be sorted by start time in descending order.

The rollout row can be expanded if the rollout is assumed to have multiple attempts (i.e., `.attempt.sequence_id > 1`). The history attempts (including the latest one) will be lazily fetched and displayed in the expanded nested table. The rollout row can also have zero attempts (e.g., when it's queuing), but it should still be displayed in the table.

The table (including the nested ones) should have the following columns:

- Rollout (from `rollout.rollout_id`)
- Attempt (from `rollout.attempts[-1].attempt_id` or "N/A" if no attempts)
- Input (from `rollout.input`, in JSON format, limit to 35 characters)
- Status (from `rollout.status`, in badge format). If it's a expanded nested row, the status should be `attempt.status`. If `rollout.status != attempt.status`, the status should be `${rollout.status} - ${attempt.status}`. Otherwise, it should be `rollout.status`.
- Resources ID (from `rollout.resources_id`)
- Mode (from `rollout.mode`)
- Start Time (if attempt is present, from `attempt.start_time`. Otherwise, from `rollout.start_time`). Time should be displayed in the local timezone as "YYYY-MM-DD HH:mm:ss".
- Duration (if attempt is present, from `attempt.end_time - attempt.start_time`. Otherwise, from `rollout`). Use a human-readable format like "1h 30m 15s".
- Last Heartbeat (use a relative time format like "1h 30m 15s" ago if the attempt is present and has a last heartbeat time. Otherwise, use "N/A").
- Worker ID (from `attempt.worker_id` or "N/A" if no attempt)
- Actions - two buttons: (1) View Raw JSON; (2) View Traces. Leave the implementation empty for now.

The implementation should only use `GET /rollouts` and `GET /rollouts/{rollout_id}/attempts` APIs. Usage of other advanced APIs like `POST /rollouts/search` is discouraged at the moment. All the pagination, sorting, filtering and searching should be handled at the frontend side.

### Settings

The page shows a list of configurations, directly affecting the Redux state and UI. No saving button is needed.

- Auto-refresh interval: can be selected from a dropdown of 5, 15, 60, 300 seconds or off. By default, it's off.
- Backend port address: can be edited as a text input. By default, it's the same as the current origin of the website.
- Theme: can be selected from a dropdown of light, dark and system default. By default, it's system default.
