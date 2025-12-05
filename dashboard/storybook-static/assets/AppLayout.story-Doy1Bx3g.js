import{a as t,j as r,T as v,S as L}from"./iframe-ByZyHG7Z.js";import{A as S,h as f,R as U,H as A}from"./AppLayout-xqSbn3cQ.js";import{c as _,P as T,i as k}from"./hooks-BqGiFhn9.js";import{c as E}from"./AppDrawer.component-BFL3lspV.js";import{S as w}from"./Stack-BYgy4ZNY.js";import"./preload-helper-PPVm8Dsz.js";import"./AppAlertBanner-yk5DN819.js";import"./IconAlertCircle-BKZdSC5f.js";import"./createReactComponent-Cbgg6bZD.js";import"./IconInfoCircle-B0aEKl0P.js";import"./table-BuDuxKCV.js";import"./find-element-ancestor-Cv-4bSct.js";import"./TracesTable.component-v8FAYs_j.js";import"./error-eSvF7V3U.js";import"./format-BWW3-KEh.js";import"./IconFileDescription-BKfQSike.js";import"./IconTimeline-BvsfcN9Y.js";const o=({title:e,description:a})=>r.jsxs(w,{gap:"sm",p:"lg",children:[r.jsx(v,{size:"lg",fw:600,children:e}),r.jsx(v,{size:"sm",c:"dimmed",children:a})]}),O=[{path:"/",element:r.jsx(S,{}),children:[{path:"rollouts",element:r.jsx(o,{title:"Rollouts",description:"Track the rollout queue and status."})},{path:"resources",element:r.jsx(o,{title:"Resources",description:"Inspect resource snapshots and metadata."})},{path:"traces",element:r.jsx(o,{title:"Traces",description:"Browse telemetry spans across attempts."})},{path:"runners",element:r.jsx(o,{title:"Runners",description:"Monitor runner activity and status."})},{path:"settings",element:r.jsx(o,{title:"Settings",description:"Configure server connection, refresh cadence, and appearance."})}]}];function R(e,a){return{alerts:[{id:"storybook-alert",message:e,tone:a,isVisible:!0,createdAt:L}]}}function s(e,a="/rollouts",y){const j=_({config:{...k,baseUrl:t},alert:y??{alerts:[]}}),b=E(O.map(x=>({...x,element:r.jsx(S,{...e})})),{initialEntries:[a]});return r.jsx(T,{store:j,children:r.jsx(U,{router:b})})}const K={title:"Layouts/AppLayout",component:S,parameters:{layout:"fullscreen"},render:e=>s(e,"/rollouts"),args:{config:{baseUrl:t,autoRefreshMs:0}}},n={args:{config:{baseUrl:"",autoRefreshMs:0}}},c={parameters:{msw:{handlers:[f.get(`${t}/v1/agl/health`,()=>A.json({status:"ok"},{status:200}))]}}},i={parameters:{msw:{handlers:[f.get(`${t}/v1/agl/health`,()=>A.json({message:"unavailable"},{status:503}))]}}},p={render:e=>s(e,"/resources")},l={render:e=>s(e,"/traces")},u={render:e=>s(e,"/settings")},m={args:{config:{baseUrl:t,autoRefreshMs:5e3}},parameters:{msw:{handlers:[f.get(`${t}/v1/agl/health`,()=>A.json({status:"ok"},{status:200}))]}}},d={render:e=>s(e,"/rollouts",R("Background synchronization completed successfully.","info"))},g={render:e=>s(e,"/rollouts",R("Rollout data may be stale. Check connectivity before proceeding.","warning"))},h={render:e=>s(e,"/rollouts",R("Unable to reach Agent-lightning API. Retry or adjust server settings.","error"))};n.parameters={...n.parameters,docs:{...n.parameters?.docs,source:{originalSource:`{
  args: {
    config: {
      baseUrl: '',
      autoRefreshMs: 0
    }
  }
}`,...n.parameters?.docs?.source}}};c.parameters={...c.parameters,docs:{...c.parameters?.docs,source:{originalSource:`{
  parameters: {
    msw: {
      handlers: [http.get(\`\${STORY_BASE_URL}/v1/agl/health\`, () => HttpResponse.json({
        status: 'ok'
      }, {
        status: 200
      }))]
    }
  }
}`,...c.parameters?.docs?.source}}};i.parameters={...i.parameters,docs:{...i.parameters?.docs,source:{originalSource:`{
  parameters: {
    msw: {
      handlers: [http.get(\`\${STORY_BASE_URL}/v1/agl/health\`, () => HttpResponse.json({
        message: 'unavailable'
      }, {
        status: 503
      }))]
    }
  }
}`,...i.parameters?.docs?.source}}};p.parameters={...p.parameters,docs:{...p.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/resources')
}`,...p.parameters?.docs?.source}}};l.parameters={...l.parameters,docs:{...l.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/traces')
}`,...l.parameters?.docs?.source}}};u.parameters={...u.parameters,docs:{...u.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/settings')
}`,...u.parameters?.docs?.source}}};m.parameters={...m.parameters,docs:{...m.parameters?.docs,source:{originalSource:`{
  args: {
    config: {
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 5000
    }
  },
  parameters: {
    msw: {
      handlers: [http.get(\`\${STORY_BASE_URL}/v1/agl/health\`, () => HttpResponse.json({
        status: 'ok'
      }, {
        status: 200
      }))]
    }
  }
}`,...m.parameters?.docs?.source}}};d.parameters={...d.parameters,docs:{...d.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/rollouts', createAlertState('Background synchronization completed successfully.', 'info'))
}`,...d.parameters?.docs?.source}}};g.parameters={...g.parameters,docs:{...g.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/rollouts', createAlertState('Rollout data may be stale. Check connectivity before proceeding.', 'warning'))
}`,...g.parameters?.docs?.source}}};h.parameters={...h.parameters,docs:{...h.parameters?.docs,source:{originalSource:`{
  render: args => renderAppLayout(args, '/rollouts', createAlertState('Unable to reach Agent-lightning API. Retry or adjust server settings.', 'error'))
}`,...h.parameters?.docs?.source}}};export{h as ErrorAlertActive,d as InfoAlertActive,n as NoServerConfigured,m as PollingEveryFiveSeconds,p as ResourcesNavActive,i as ServerOffline,c as ServerOnline,u as SettingsNavActive,l as TracesNavActive,g as WarningAlertActive,K as default};
