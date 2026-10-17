"use client";

import * as React from "react";

/**
 * Data-fetching hook for page-level API loads.
 *
 * Why this exists rather than calling `load()` straight from an effect:
 *
 *  - No synchronous setState in the effect body. Every state update happens
 *    after an await, which avoids the cascading re-render the React Compiler
 *    warns about (`react-hooks/set-state-in-effect`).
 *  - Cancellation. If the user navigates away mid-request the response is
 *    discarded instead of setting state on an unmounted component — a real
 *    problem here, because inference requests can take a second or two.
 *  - One consistent loading/error/retry shape across every page.
 */
export function useAsyncData<T>(
  fetcher: () => Promise<T>,
  deps: React.DependencyList = [],
) {
  const [data, setData] = React.useState<T | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [nonce, setNonce] = React.useState(0);

  // Keep the latest fetcher without making it an effect dependency, so callers
  // can pass an inline closure without triggering a refetch loop.
  const ref = React.useRef(fetcher);
  React.useEffect(() => { ref.current = fetcher; });

  React.useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const result = await ref.current();
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Request failed");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = React.useCallback(() => {
    setLoading(true);
    setNonce((n) => n + 1);
  }, []);

  return { data, loading, error, reload, setData } as const;
}
