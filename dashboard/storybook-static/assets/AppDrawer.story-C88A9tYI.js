import{j as h,b as T,a as g}from"./iframe-ByZyHG7Z.js";import{c as b,r as w,P as D,i as N,d as j,e as A,a as _}from"./hooks-BqGiFhn9.js";import{A as S}from"./AppDrawer.component-BFL3lspV.js";import"./preload-helper-PPVm8Dsz.js";import"./format-BWW3-KEh.js";import"./TracesTable.component-v8FAYs_j.js";import"./table-BuDuxKCV.js";import"./createReactComponent-Cbgg6bZD.js";import"./Stack-BYgy4ZNY.js";import"./find-element-ancestor-Cv-4bSct.js";import"./error-eSvF7V3U.js";import"./IconAlertCircle-BKZdSC5f.js";import"./IconFileDescription-BKfQSike.js";const H={title:"Components/AppDrawer",component:S,parameters:{layout:"fullscreen"}},t=T,R={rolloutId:"ro-story-001",attemptId:"at-story-001",sequenceId:1,startTime:t-3600,endTime:null,status:"running",workerId:"worker-story",lastHeartbeatTime:t-42,metadata:{info:"Sample metadata",runId:"run-123"}},e={rolloutId:"ro-story-001",input:{task:"Generate daily summary",payload:{account:"enterprise",date:"2024-02-19"}},startTime:t-4e3,endTime:null,mode:"train",resourcesId:"rs-story-001",status:"running",config:{retries:1,priority:"high"},metadata:{owner:"storybook"},attempt:R},x={...e,status:"queuing",attempt:null},s={...e,status:"running",attempt:{...R,status:"failed",endTime:t-1200,metadata:{info:"Latest attempt failed",reason:"Timeout"}}},a={rolloutId:"ro-story-001",attemptId:"at-story-001",sequenceId:2,traceId:"tr-story-001",spanId:"sp-story-001",parentId:null,name:"Fetch Resources",status:{status_code:"OK",description:"Completed successfully"},attributes:{"http.method":"GET","http.url":"https://api.example.com/resources",duration_ms:120},startTime:t-240,endTime:t-120,events:[],links:[],context:{},parent:null,resource:{}},O=[a,{...a,spanId:"sp-story-002",name:"Process Response",parentId:"sp-story-001",sequenceId:3,status:{status_code:"ERROR",description:"Unexpected response code"},attributes:{...a.attributes,duration_ms:240},startTime:t-120,endTime:t-30}];function r(o,y){const f=b({config:{...N,baseUrl:g},rollouts:_,resources:A,traces:j,drawer:{isOpen:!0,content:o}});if(o.type==="rollout-traces"&&y?.spans){const I={rolloutId:o.rollout.rolloutId,attemptId:o.attempt?.attemptId??void 0,limit:100,offset:0,sortBy:"start_time",sortOrder:"desc"};f.dispatch(w.util.upsertQueryData("getSpans",I,{items:y.spans,total:y.spans.length,limit:100,offset:0}))}return h.jsx(D,{store:f,children:h.jsx(S,{})})}const n={render:()=>r({type:"rollout-json",rollout:e,attempt:e.attempt,isNested:!1})},l={render:()=>r({type:"rollout-json",rollout:e,attempt:{...R,attemptId:"at-story-002",sequenceId:2,status:"failed",endTime:t-1200,metadata:{info:"Secondary attempt",reason:"Timeout"}},isNested:!0})},p={render:()=>r({type:"rollout-traces",rollout:e,attempt:e.attempt,isNested:!1},{spans:O})},m={render:()=>r({type:"rollout-json",rollout:x,attempt:null,isNested:!1})},i={render:()=>r({type:"rollout-json",rollout:s,attempt:s.attempt,isNested:!1})},u={render:()=>r({type:"trace-detail",span:a,rollout:s,attempt:s.attempt})},d={render:()=>r({type:"rollout-json",rollout:e,attempt:e.attempt,isNested:!1}),parameters:{theme:"light"}},c={render:()=>r({type:"trace-detail",span:{...a,spanId:"sp-story-002",name:"Process Response",status:{status_code:"ERROR",description:"Unexpected response code"}},rollout:s,attempt:s.attempt}),parameters:{theme:"dark"}};n.parameters={...n.parameters,docs:{...n.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-json',
    rollout: baseRollout,
    attempt: baseRollout.attempt,
    isNested: false
  })
}`,...n.parameters?.docs?.source}}};l.parameters={...l.parameters,docs:{...l.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-json',
    rollout: baseRollout,
    attempt: {
      ...baseAttempt,
      attemptId: 'at-story-002',
      sequenceId: 2,
      status: 'failed',
      endTime: now - 1200,
      metadata: {
        info: 'Secondary attempt',
        reason: 'Timeout'
      }
    },
    isNested: true
  })
}`,...l.parameters?.docs?.source}}};p.parameters={...p.parameters,docs:{...p.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-traces',
    rollout: baseRollout,
    attempt: baseRollout.attempt,
    isNested: false
  }, {
    spans: sampleTraces
  })
}`,...p.parameters?.docs?.source}}};m.parameters={...m.parameters,docs:{...m.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-json',
    rollout: noAttemptRollout,
    attempt: null,
    isNested: false
  })
}`,...m.parameters?.docs?.source}}};i.parameters={...i.parameters,docs:{...i.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-json',
    rollout: mismatchRollout,
    attempt: mismatchRollout.attempt,
    isNested: false
  })
}`,...i.parameters?.docs?.source}}};u.parameters={...u.parameters,docs:{...u.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'trace-detail',
    span: sampleSpan,
    rollout: mismatchRollout,
    attempt: mismatchRollout.attempt
  })
}`,...u.parameters?.docs?.source}}};d.parameters={...d.parameters,docs:{...d.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'rollout-json',
    rollout: baseRollout,
    attempt: baseRollout.attempt,
    isNested: false
  }),
  parameters: {
    theme: 'light'
  }
}`,...d.parameters?.docs?.source}}};c.parameters={...c.parameters,docs:{...c.parameters?.docs,source:{originalSource:`{
  render: () => renderWithDrawer({
    type: 'trace-detail',
    span: {
      ...sampleSpan,
      spanId: 'sp-story-002',
      name: 'Process Response',
      status: {
        status_code: 'ERROR',
        description: 'Unexpected response code'
      }
    },
    rollout: mismatchRollout,
    attempt: mismatchRollout.attempt
  }),
  parameters: {
    theme: 'dark'
  }
}`,...c.parameters?.docs?.source}}};export{c as DarkTheme,d as LightTheme,l as NestedAttemptJson,m as NoAttempt,n as RolloutJson,p as RolloutTraces,u as SpanDetail,i as StatusMismatch,H as default};
