import { AppShell, Badge, Group, Image, NavLink as MantineNavLink, Stack, Text } from '@mantine/core';
import { type ReactNode, useEffect, useMemo, useState } from 'react';
import { NavLink as RouterNavLink, Outlet, useLocation } from 'react-router-dom';
import { selectConfig } from '../features/config';
import { AppDrawer } from '@/components/AppDrawer.component';
import { AppAlertBanner } from '@/components/AppAlertBanner';
import { useAppSelector } from '../store/hooks';
import faviconUrl from '../favicon.svg';
import { IconTimeline, IconCpu, IconRouteSquare, IconSettings } from '@tabler/icons-react';

type ConnectionStatus = 'online' | 'offline' | 'unknown';

type NavItem = {
  label: string;
  to: string;
  icon: ReactNode;
};

const NAV_ITEMS: NavItem[] = [
  { label: 'Rollouts', to: '/rollouts', icon: <IconRouteSquare size={16} /> },
  { label: 'Resources', to: '/resources', icon: <IconCpu size={16} /> },
  { label: 'Traces', to: '/traces', icon: <IconTimeline size={16} /> },
  { label: 'Settings', to: '/settings', icon: <IconSettings size={16} /> },
];

const CONNECTION_STATUS_META: Record<ConnectionStatus, { color: string; label: string }> = {
  offline: { color: 'red', label: 'Offline' },
  online: { color: 'teal', label: 'Online' },
  unknown: { color: 'gray', label: 'Unknown' },
};

const DEFAULT_AUTO_REFRESH_MS = 30_000;

type ConnectionOptions = {
  baseUrl?: string;
  autoRefreshMs?: number;
};

function getSameOriginUrl() {
  return window.location.origin;
}

function buildHealthUrl(baseUrl: string) {
  if (baseUrl.startsWith('http://') || baseUrl.startsWith('https://')) {
    const normalized = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    return new URL('health', normalized).toString();
  }

  const normalizedBase = baseUrl.startsWith('/') ? baseUrl : `/${baseUrl}`;
  const trimmed = normalizedBase.replace(/\/+$/, '');
  return `${trimmed}/health`;
}

function useServerConnection({ baseUrl, autoRefreshMs }: ConnectionOptions) {
  const [status, setStatus] = useState<ConnectionStatus>('unknown');
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (!baseUrl) {
      setStatus('unknown');
      setIsRefreshing(false);
      return;
    }

    let disposed = false;
    let intervalId: number | undefined;
    let activeController: AbortController | undefined;
    const healthUrl = buildHealthUrl(baseUrl);

    const check = async () => {
      activeController?.abort();
      const controller = new AbortController();
      activeController = controller;
      if (!disposed) {
        setIsRefreshing(true);
      }

      try {
        const response = await fetch(healthUrl, { signal: controller.signal });
        if (!disposed && activeController === controller) {
          setStatus(response.ok ? 'online' : 'offline');
        }
      } catch (error) {
        if (disposed || (error instanceof DOMException && error.name === 'AbortError')) {
          return;
        }

        if (!disposed && activeController === controller) {
          setStatus('offline');
        }
      } finally {
        if (!disposed && activeController === controller) {
          setIsRefreshing(false);
        }
      }
    };

    check();

    if (autoRefreshMs && autoRefreshMs > 0) {
      intervalId = window.setInterval(check, autoRefreshMs);
    }

    return () => {
      disposed = true;
      activeController?.abort();
      if (intervalId) {
        window.clearInterval(intervalId);
      }
    };
  }, [autoRefreshMs, baseUrl]);

  return { status, isRefreshing };
}

function ConnectionIndicator({
  baseUrl,
  status = 'unknown',
  isRefreshing = false,
}: {
  baseUrl?: string;
  status?: ConnectionStatus;
  isRefreshing?: boolean;
}) {
  const { color, label } = CONNECTION_STATUS_META[status ?? 'unknown'];
  const connectionTarget = baseUrl && baseUrl.length > 0 ? baseUrl : 'No server configured';

  return (
    <Stack gap={4} data-testid="connection-indicator">
      <Text size="xs" fw={600} c="dimmed">
        Server connection
      </Text>
      <Group gap="xs">
        {isRefreshing ? (
          <Badge color="blue" variant="light" radius="xl" className="connection-refreshing">
            Refreshing
          </Badge>
        ) : (
          <Badge color={color} variant="light" radius="xl">
            {label}
          </Badge>
        )}
        <Text size="xs" c="dimmed">
          {connectionTarget}
        </Text>
      </Group>
    </Stack>
  );
}

export type AppLayoutProps = {
  config?: ConnectionOptions;
};

export function AppLayout({ config }: AppLayoutProps = {}) {
  const location = useLocation();
  const resolvedBaseUrl = config?.baseUrl ?? getSameOriginUrl() ?? '';
  const autoRefreshMs =
    config?.autoRefreshMs !== undefined ? config.autoRefreshMs : DEFAULT_AUTO_REFRESH_MS;
  const connectionState = useServerConnection({
    baseUrl: resolvedBaseUrl || undefined,
    autoRefreshMs,
  });
  const navItems = useMemo(
    () =>
      NAV_ITEMS.map((item) => ({
        ...item,
        active: location.pathname === item.to || location.pathname.startsWith(`${item.to}/`),
      })),
    [location.pathname],
  );

  return (
    <AppShell padding="md" navbar={{ width: 280, breakpoint: 'sm' }}>
      <AppShell.Navbar>
        <AppShell.Section p="md" mb="md">
          <Group gap="sm">
            <Image src={faviconUrl} alt="Agent-lightning logo" w={32} h={32} />
            <Text fw={600} size="sm">
              Agent-lightning Dashboard
            </Text>
          </Group>
        </AppShell.Section>
        <AppShell.Section grow p="md" pt={0}>
          <Stack gap="xs">
            {navItems.map((item) => (
              <MantineNavLink
                key={item.to}
                component={RouterNavLink}
                to={item.to}
                label={item.label}
                active={item.active}
                variant="light"
                leftSection={item.icon}
              />
            ))}
          </Stack>
        </AppShell.Section>
        <AppShell.Section p="md">
          <ConnectionIndicator
            baseUrl={resolvedBaseUrl || undefined}
            status={connectionState.status}
            isRefreshing={connectionState.isRefreshing}
          />
        </AppShell.Section>
      </AppShell.Navbar>
      <AppShell.Main>
        <AppAlertBanner />
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
}

export function AppLayoutWithState() {
  const config = useAppSelector(selectConfig);

  return (
    <>
      <AppLayout
        config={{
          baseUrl: config.baseUrl,
          autoRefreshMs: config.autoRefreshMs,
        }}
      />
      <AppDrawer />
    </>
  );
}
