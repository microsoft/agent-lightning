// Copyright (c) Microsoft. All rights reserved.

import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom';
import { AppLayoutWithState } from './layouts/AppLayout';
import { ResourcesPage } from './pages/Resources.page';
import { RolloutsPage } from './pages/Rollouts.page';
import { SettingsPage } from './pages/Settings.page';
import { TracesPage } from './pages/Traces.page';
import { WorkersPage } from './pages/Workers.page';

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayoutWithState />,
    children: [
      {
        index: true,
        element: <Navigate to='/rollouts' replace />,
      },
      {
        path: 'rollouts',
        element: <RolloutsPage />,
      },
      {
        path: 'resources',
        element: <ResourcesPage />,
      },
      {
        path: 'traces',
        element: <TracesPage />,
      },
      {
        path: 'runners',
        element: <WorkersPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
    ],
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
