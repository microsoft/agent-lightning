// Copyright (c) Microsoft. All rights reserved.

import { useEffect, useState } from 'react';
import { IconAlertCircle, IconAlertTriangle, IconInfoCircle } from '@tabler/icons-react';
import { Notification, Portal, Transition } from '@mantine/core';
import { hideAlert, selectHighestPriorityAlert, type AppAlert } from '@/features/ui/alert';
import { useAppDispatch, useAppSelector } from '@/store/hooks';

const ALERT_META = {
  info: {
    color: 'blue',
    icon: IconInfoCircle,
  },
  warning: {
    color: 'yellow',
    icon: IconAlertTriangle,
  },
  error: {
    color: 'red',
    icon: IconAlertCircle,
  },
} as const;

export function AppAlertBanner() {
  const dispatch = useAppDispatch();
  const alert = useAppSelector(selectHighestPriorityAlert);
  const [transitionAlert, setTransitionAlert] = useState<AppAlert | null>(alert);

  useEffect(() => {
    if (alert) {
      setTransitionAlert(alert);
    }
  }, [alert]);

  const handleClose = (id?: string) => {
    if (id) {
      dispatch(hideAlert({ id }));
    }
  };

  const currentAlert = alert ?? transitionAlert;
  const mounted = Boolean(alert);

  if (!currentAlert) {
    return null;
  }

  const meta = ALERT_META[currentAlert.tone];
  const IconComponent = meta.icon;

  return (
    <Portal>
      <Transition
        mounted={mounted}
        transition='slide-down'
        duration={200}
        timingFunction='ease'
        onExited={() => setTransitionAlert(null)}
      >
        {(styles) => (
          <Notification
            icon={<IconComponent size={18} />}
            color={meta.color}
            variant='light'
            withCloseButton
            onClose={() => handleClose(currentAlert.id)}
            style={{
              position: 'fixed',
              top: 16,
              right: 16,
              maxWidth: 450,
              width: 'calc(100% - 32px)',
              zIndex: 2000,
              boxShadow: 'var(--mantine-shadow-md)',
              ...styles,
            }}
          >
            {currentAlert.message}
          </Notification>
        )}
      </Transition>
    </Portal>
  );
}
