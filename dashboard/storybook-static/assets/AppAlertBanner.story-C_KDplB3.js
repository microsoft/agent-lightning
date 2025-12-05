import{j as e,S,a as l}from"./iframe-ByZyHG7Z.js";import{c,P as d,i as p,a as u,b as m}from"./hooks-BqGiFhn9.js";import{A as s}from"./AppAlertBanner-yk5DN819.js";import"./preload-helper-PPVm8Dsz.js";import"./format-BWW3-KEh.js";import"./IconAlertCircle-BKZdSC5f.js";import"./createReactComponent-Cbgg6bZD.js";import"./IconInfoCircle-B0aEKl0P.js";const U={title:"Components/AppAlertBanner",component:s,parameters:{layout:"fullscreen"}};function i(a,g){const A={alerts:[{id:"storybook-alert",message:a,tone:g,isVisible:!0,createdAt:S}]},h=c({config:{...p,baseUrl:l},drawer:m,rollouts:u,alert:A});return e.jsx(d,{store:h,children:e.jsx("div",{style:{padding:24},children:e.jsx(s,{})})})}const r={render:()=>i("Background synchronization completed successfully.","info")},t={render:()=>i("Rollout data may be stale. Check your network connection before continuing.","warning")},n={render:()=>i("Unable to reach the Agent-lightning API. Retry or adjust the backend settings.","error")},o={render:()=>{const a=c({config:{...p,baseUrl:l},drawer:m,rollouts:u,alert:{alerts:[]}});return e.jsx(d,{store:a,children:e.jsx("div",{style:{padding:24},children:e.jsx(s,{})})})}};r.parameters={...r.parameters,docs:{...r.parameters?.docs,source:{originalSource:`{
  render: () => renderWithAlert('Background synchronization completed successfully.', 'info')
}`,...r.parameters?.docs?.source}}};t.parameters={...t.parameters,docs:{...t.parameters?.docs,source:{originalSource:`{
  render: () => renderWithAlert('Rollout data may be stale. Check your network connection before continuing.', 'warning')
}`,...t.parameters?.docs?.source}}};n.parameters={...n.parameters,docs:{...n.parameters?.docs,source:{originalSource:`{
  render: () => renderWithAlert('Unable to reach the Agent-lightning API. Retry or adjust the backend settings.', 'error')
}`,...n.parameters?.docs?.source}}};o.parameters={...o.parameters,docs:{...o.parameters?.docs,source:{originalSource:`{
  render: () => {
    const store = createAppStore({
      config: {
        ...initialConfigState,
        baseUrl: STORY_BASE_URL
      },
      drawer: initialDrawerState,
      rollouts: initialRolloutsUiState,
      alert: {
        alerts: []
      }
    });
    return <Provider store={store}>
        <div style={{
        padding: 24
      }}>
          <AppAlertBanner />
        </div>
      </Provider>;
  }
}`,...o.parameters?.docs?.source}}};export{n as ErrorAlert,r as InfoAlert,o as NoAlert,t as WarningAlert,U as default};
