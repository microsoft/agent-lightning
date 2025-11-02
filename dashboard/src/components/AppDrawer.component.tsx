import { useMemo } from 'react';
import {
  ActionIcon,
  Badge,
  Box,
  CopyButton,
  Drawer,
  Group,
  Stack,
  Text,
  Tooltip,
  useMantineColorScheme,
} from '@mantine/core';
import { IconCheck, IconCopy } from '@tabler/icons-react';
import { Editor } from '@monaco-editor/react';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { closeDrawer, selectDrawerContent, selectDrawerIsOpen } from '@/features/ui/drawer';
import { formatStatusLabel } from '@/utils/format';
import type { AttemptStatus, RolloutStatus, Span } from '@/types';

const ATTEMPT_STATUS_COLORS: Record<AttemptStatus, string> = {
  failed: 'red',
  preparing: 'violet',
  running: 'blue',
  succeeded: 'teal',
  timeout: 'orange',
  unresponsive: 'orange',
};

const ROLLOUT_STATUS_COLORS: Record<RolloutStatus, string> = {
  cancelled: 'gray',
  failed: 'red',
  preparing: 'violet',
  queuing: 'blue',
  requeuing: 'cyan',
  running: 'blue',
  succeeded: 'teal',
};

const SPAN_STATUS_COLORS: Record<Span['status']['status_code'], string> = {
  UNSET: 'gray',
  OK: 'teal',
  ERROR: 'red',
};

function getStatusBadgeColor(status: RolloutStatus | AttemptStatus, isAttempt: boolean) {
  if (isAttempt) {
    return ATTEMPT_STATUS_COLORS[status as AttemptStatus] ?? 'gray';
  }

  return ROLLOUT_STATUS_COLORS[status as RolloutStatus] ?? 'gray';
}

function formatJson(value: unknown) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function AppDrawer() {
  const dispatch = useAppDispatch();
  const isOpen = useAppSelector(selectDrawerIsOpen);
  const content = useAppSelector(selectDrawerContent);
  const { colorScheme } = useMantineColorScheme();

  const handleClose = () => {
    dispatch(closeDrawer());
  };

  const editorTheme = colorScheme === 'dark' ? 'vs-dark' : 'vs-light';

  const derivedContent = useMemo(() => {
    if (!content) {
      return null;
    }

    if (content.type === 'trace-detail') {
      const span = content.span;
      const spanStatusCode = span.status?.status_code ?? null;
      const spanBadgeColor = spanStatusCode ? SPAN_STATUS_COLORS[spanStatusCode] ?? 'gray' : undefined;
      const formattedJson = formatJson(span);

      const title = (
        <Stack gap={3}>
          <Group gap={6}>
            <Text fw={600}>{span.name ?? span.spanId}</Text>
            {spanStatusCode
              ? (
                  <Badge size='sm' variant='light' color={spanBadgeColor}>
                    {spanStatusCode}
                  </Badge>
                )
              : null}
          </Group>
          <Group gap={6}>
            <Text size='sm' c='dimmed'>
              {span.spanId}
            </Text>
            <CopyButton value={span.spanId}>
              {({ copied, copy }) => (
                <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
                  <ActionIcon
                    aria-label={`Copy span ID ${span.spanId}`}
                    variant='subtle'
                    color={copied ? 'teal' : 'gray'}
                    size='sm'
                    onClick={(event) => {
                      event.stopPropagation();
                      copy();
                    }}
                  >
                    {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                  </ActionIcon>
                </Tooltip>
              )}
            </CopyButton>
          </Group>
          <Group gap='xs'>
            <Group gap={3}>
              <Text size='sm' c='dimmed' fw={500}>
                Rollout
              </Text>
              <Text size='sm' c='dimmed'>
                {span.rolloutId}
              </Text>
            </Group>
            <Group gap={3}>
              <Text size='sm' c='dimmed' fw={500}>
                Attempt
              </Text>
              <Text size='sm' c='dimmed'>
                {span.attemptId ?? '—'}
              </Text>
            </Group>
          </Group>
        </Stack>
      );

      return {
        titleContent: title,
        heading: span.name ?? span.spanId,
        bodyContent: (
          <Box style={{ flex: 1, minHeight: 0 }}>
            <Editor
              height='100%'
              language='json'
              value={formattedJson}
              theme={editorTheme}
              options={{
                readOnly: true,
                domReadOnly: true,
                minimap: { enabled: false },
                automaticLayout: true,
                scrollBeyondLastLine: false,
                fontSize: 13,
              }}
            />
          </Box>
        ),
      };
    }

    const rollout = content.rollout;
    const attempt = content.attempt;
    const rolloutId = rollout.rolloutId;
    const attemptId = attempt?.attemptId ?? null;
    const rolloutStatus = rollout.status ?? null;
    const attemptStatus = attempt?.status ?? null;
    const rolloutStatusLabel = rolloutStatus ? formatStatusLabel(rolloutStatus) : null;
    const attemptStatusLabel = attemptStatus ? formatStatusLabel(attemptStatus) : null;
    const hasStatusMismatch
      = rolloutStatus !== null && attemptStatus !== null && rolloutStatus !== attemptStatus;
    const rolloutBadgeColor = rolloutStatus ? getStatusBadgeColor(rolloutStatus, false) : undefined;
    const attemptBadgeColor = attemptStatus ? getStatusBadgeColor(attemptStatus, true) : undefined;
    const showRolloutBadgeInHeading = Boolean(rolloutStatusLabel && (!attemptStatus || hasStatusMismatch));
    const showAttemptBadge = Boolean(attemptStatusLabel && attemptStatus);

    const title = (
      <Stack gap={3}>
        <Group gap={6}>
          <Text fw={600}>{rolloutId}</Text>
          <CopyButton value={rolloutId}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
                <ActionIcon
                  aria-label={`Copy rollout ID ${rolloutId}`}
                  variant='subtle'
                  color={copied ? 'teal' : 'gray'}
                  size='sm'
                  onClick={(event) => {
                    event.stopPropagation();
                    copy();
                  }}
                >
                  {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                </ActionIcon>
              </Tooltip>
            )}
          </CopyButton>
          {showRolloutBadgeInHeading && rolloutStatusLabel
            ? (
                <Badge size='sm' variant='light' color={rolloutBadgeColor}>
                  {rolloutStatusLabel}
                </Badge>
              )
            : null}
        </Group>
        <Group gap='xs'>
          {attemptId
            ? (
                <Group gap={3}>
                  <Text size='sm' c='dimmed' fw={500}>
                    Attempt
                  </Text>
                  <Text size='sm' c='dimmed'>
                    {attemptId}
                  </Text>
                </Group>
              )
            : null}
          {showAttemptBadge && attemptStatusLabel
            ? (
                <Badge size='sm' variant='light' color={attemptBadgeColor}>
                  {attemptStatusLabel}
                </Badge>
              )
            : null}
          {!showRolloutBadgeInHeading && !attemptStatus && rolloutStatusLabel
            ? (
                <Badge size='sm' variant='light' color={rolloutBadgeColor}>
                  {rolloutStatusLabel}
                </Badge>
              )
            : null}
        </Group>
      </Stack>
    );

    const jsonValue
      = content.type === 'rollout-json'
        ? content.isNested && content.attempt
          ? content.attempt
          : content.rollout
        : null;

    const body
      = jsonValue !== null
        ? (
            <Box style={{ flex: 1, minHeight: 0 }}>
              <Editor
                height='100%'
                language='json'
                value={formatJson(jsonValue)}
                theme={editorTheme}
                options={{
                  readOnly: true,
                  domReadOnly: true,
                  minimap: { enabled: false },
                  automaticLayout: true,
                  scrollBeyondLastLine: false,
                  fontSize: 13,
                }}
              />
            </Box>
          )
        : null;

    return {
      titleContent: title,
      heading: rolloutId,
      bodyContent: body,
    };
  }, [content, editorTheme]);

  if (!content || !derivedContent) {
    return null;
  }

  const { titleContent, heading, bodyContent } = derivedContent;

  return (
    <Drawer
      position='right'
      size='lg'
      opened={isOpen}
      onClose={handleClose}
      overlayProps={{ opacity: 0.5 }}
      withinPortal
      styles={{
        content: {
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '100vh',
        },
        body: {
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          padding: 'var(--mantine-spacing-md)',
          minHeight: 0,
          overflow: 'hidden',
        },
      }}
      title={titleContent ?? heading}
    >
      <Stack gap='md' h='100%' style={{ flex: 1, minHeight: 0 }}>
        {bodyContent}
      </Stack>
    </Drawer>
  );
}
