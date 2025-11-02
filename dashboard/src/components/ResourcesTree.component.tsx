// Copyright (c) Microsoft. All rights reserved.

import { useMemo } from 'react';
import { IconAlertCircle, IconChevronRight } from '@tabler/icons-react';
import { Box, Group, Stack, Text, Tree, type TreeNodeData } from '@mantine/core';
import type { Resources } from '@/types';
import { safeStringify } from '@/utils/format';

function convertToTreeData(obj: any, key: string = 'root', parentPath = ''): TreeNodeData {
  const isObject = obj !== null && typeof obj === 'object' && !Array.isArray(obj);
  const isArray = Array.isArray(obj);
  const currentPath = parentPath ? `${parentPath}.${key}` : key;

  if (isObject) {
    const children = Object.entries(obj).map(([childKey, childValue]) =>
      convertToTreeData(childValue, childKey, currentPath),
    );

    return {
      value: currentPath,
      label: (
        <Group gap={6}>
          <Text size='sm' fw={500}>
            {key}
          </Text>
          <Text size='xs' c='dimmed'>
            (Object)
          </Text>
        </Group>
      ),
      children: children.length > 0 ? children : undefined,
    };
  }

  if (isArray) {
    const children = obj.map((item: any, index: number) => convertToTreeData(item, `[${index}]`, currentPath));

    return {
      value: currentPath,
      label: (
        <Group gap={6}>
          <Text size='sm' fw={500}>
            {key}
          </Text>
          <Text size='xs' c='dimmed'>
            (Array[
            {obj.length}
            ])
          </Text>
        </Group>
      ),
      children: children.length > 0 ? children : undefined,
    };
  }

  // Primitive value
  return {
    value: currentPath,
    label: (
      <Group gap={6}>
        <Text size='sm' fw={500}>
          {key}:
        </Text>
        <Text size='sm' ff='monospace' c='dimmed'>
          {safeStringify(obj)}
        </Text>
      </Group>
    ),
  };
}

export type ResourcesTreeProps = {
  resources: Resources;
};

export function ResourcesTree({ resources }: ResourcesTreeProps) {
  const resourcesDict = resources.resources ?? {};

  const treeData = useMemo<TreeNodeData[]>(() => {
    const entries = Object.entries(resourcesDict);

    if (entries.length === 0) {
      return [];
    }

    return entries.map(([key, value]) => convertToTreeData(value, key));
  }, [resourcesDict]);

  if (treeData.length === 0) {
    return (
      <Stack gap='xs' align='center' py='md'>
        <IconAlertCircle size={24} color='gray' />
        <Text size='sm' c='dimmed'>
          No resources found
        </Text>
      </Stack>
    );
  }

  return (
    <Box p='md' style={{ backgroundColor: 'var(--mantine-color-default-hover)' }}>
      <Tree
        data={treeData}
        levelOffset={20}
        expandOnClick
        selectOnClick
        renderNode={({ node, expanded, hasChildren, elementProps }) => (
          <Group gap={4} {...elementProps}>
            {hasChildren && (
              <Box
                style={{
                  minWidth: 14,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
                  transition: 'transform 150ms ease',
                }}
              >
                <IconChevronRight size={14} />
              </Box>
            )}
            {node.label}
          </Group>
        )}
      />
    </Box>
  );
}
