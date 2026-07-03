// Shared backoffice navigation, so every admin page can move between the queue, quality, health,
// and insights views.
export function AdminNav() {
  return (
    <nav>
      <a href="/admin">Queue</a>
      <a href="/admin/quality">Quality</a>
      <a href="/admin/health">Health</a>
      <a href="/admin/insights">Insights</a>
    </nav>
  );
}
