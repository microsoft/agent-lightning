import{r as l,j as a,a as $,b as Q}from"./iframe-ByZyHG7Z.js";import{S as z,a as J,e as R,g as b,u as W,w as A}from"./modes-Di9-PqFx.js";import{u as Y,f as i,a8 as G,a9 as K,aa as V,ab as X,ac as Z,ad as ee,ae as re,af as te,ag as ae,ah as se,ai as ne,L as oe,t as le,v as x,P as D,c as ie,i as ce,aj as ue}from"./hooks-BqGiFhn9.js";import{A as de,R as me}from"./AppLayout-xqSbn3cQ.js";import{A as pe}from"./AppAlertBanner-yk5DN819.js";import{c as he,A as v}from"./AppDrawer.component-BFL3lspV.js";import{W as we}from"./WorkersTable.component-D7qGphi2.js";import{s as ke}from"./selectors-5GhMe3AS.js";import{g as ye}from"./error-eSvF7V3U.js";import{S as fe}from"./Stack-BYgy4ZNY.js";import{T as Te}from"./Title-Czq4HMLJ.js";import{T as ge}from"./TextInput-Ba5dtmf_.js";import{I as Se}from"./IconSearch-Dks1vVM2.js";import"./preload-helper-PPVm8Dsz.js";import"./format-BWW3-KEh.js";import"./table-BuDuxKCV.js";import"./createReactComponent-Cbgg6bZD.js";import"./find-element-ancestor-Cv-4bSct.js";import"./TracesTable.component-v8FAYs_j.js";import"./IconAlertCircle-BKZdSC5f.js";import"./IconFileDescription-BKfQSike.js";import"./IconTimeline-BvsfcN9Y.js";import"./IconInfoCircle-B0aEKl0P.js";function w(){const r=Y(),t=i(ke),n=i(G),o=i(K),B=i(V),q=i(X),P=i(Z),I=ee(P,{pollingInterval:t>0?t:void 0}),c=I.data,{isLoading:f,isFetching:T,isError:g,error:S,refetch:j}=I,C=l.useCallback(s=>{r(re(s))},[r]),H=l.useCallback(s=>{r(te({column:s.columnAccessor,direction:s.direction}))},[r]),F=l.useCallback(s=>{r(ae(s))},[r]),_=l.useCallback(s=>{r(se(s))},[r]),M=l.useCallback(()=>{r(ne())},[r]),U=l.useCallback(s=>{r(oe({type:"worker-detail",worker:s}))},[r]),O=Array.isArray(c?.items)&&c.items.length>0,L=f&&!O;return l.useEffect(()=>{if(g){const s=ye(S),N=s?` (${s})`:"";r(le({id:"workers-fetch",message:`Unable to refresh workers${N}. The table may be out of date until the connection recovers.`,tone:"error"}));return}!f&&!T&&r(x({id:"workers-fetch"}))},[r,S,g,T,f]),l.useEffect(()=>()=>{r(x({id:"workers-fetch"}))},[r]),a.jsxs(fe,{gap:"md",children:[a.jsx(Te,{order:1,children:"Runners"}),a.jsx(ge,{placeholder:"Search by Runner ID",value:n,onChange:s=>C(s.currentTarget.value),leftSection:a.jsx(Se,{size:16}),"data-testid":"workers-search-input",w:"100%",style:{maxWidth:360}}),L?a.jsx(z,{height:360,radius:"md"}):a.jsx(we,{workers:c?.items,totalRecords:c?.total??0,isFetching:T,isError:g,error:S,searchTerm:n,sort:q,page:o,recordsPerPage:B,onSortStatusChange:H,onPageChange:F,onRecordsPerPageChange:_,onResetFilters:M,onRefetch:j,onShowDetails:U})]})}w.__docgenInfo={description:"",methods:[],displayName:"WorkersPage"};const Ye={title:"Pages/WorkersPage",component:w,parameters:{layout:"fullscreen",chromatic:{modes:J}}},e=Q,be=[{workerId:"worker-east",status:"busy",heartbeatStats:{queueDepth:2,gpuUtilization:.82},lastHeartbeatTime:e-20,lastDequeueTime:e-120,lastBusyTime:e-60,lastIdleTime:e-600,currentRolloutId:"ro-story-001",currentAttemptId:"at-story-010"},{workerId:"worker-west",status:"busy",heartbeatStats:{queueDepth:1},lastHeartbeatTime:e-45,lastDequeueTime:e-300,lastBusyTime:e-120,lastIdleTime:e-4800,currentRolloutId:"ro-story-003",currentAttemptId:"at-story-033"},{workerId:"worker-north",status:"idle",heartbeatStats:{queueDepth:0},lastHeartbeatTime:e-120,lastDequeueTime:e-3600,lastBusyTime:e-5400,lastIdleTime:e-180,currentRolloutId:null,currentAttemptId:null},{workerId:"worker-south",status:"idle",heartbeatStats:null,lastHeartbeatTime:e-900,lastDequeueTime:e-7200,lastBusyTime:e-8600,lastIdleTime:e-8600,currentRolloutId:null,currentAttemptId:null},{workerId:"worker-central",status:"busy",heartbeatStats:{queueDepth:3,cpuUtilization:.55},lastHeartbeatTime:e-8,lastDequeueTime:e-45,lastBusyTime:e-10,lastIdleTime:e-900,currentRolloutId:"ro-story-005",currentAttemptId:"at-story-013"},{workerId:"worker-standby",status:"idle",heartbeatStats:{queueDepth:0,threads:32},lastHeartbeatTime:e-300,lastDequeueTime:e-10800,lastBusyTime:e-14400,lastIdleTime:e-200,currentRolloutId:null,currentAttemptId:null},{workerId:"worker-observer",status:"unknown",heartbeatStats:{queueDepth:0},lastHeartbeatTime:e-30,lastDequeueTime:e-6400,lastBusyTime:null,lastIdleTime:null,currentRolloutId:null,currentAttemptId:null}],k=R(be);function E(r){return ie({config:{...ce,baseUrl:$,autoRefreshMs:0,...r},workers:ue})}function y(r){const t=E(r);return a.jsx(D,{store:t,children:a.jsxs(a.Fragment,{children:[a.jsx(w,{}),a.jsx(pe,{}),a.jsx(v,{})]})})}function Ie(r){const t=E(r),n=he([{path:"/",element:a.jsx(de,{config:{baseUrl:t.getState().config.baseUrl,autoRefreshMs:t.getState().config.autoRefreshMs}}),children:[{path:"/runners",element:a.jsx(w,{})}]}],{initialEntries:["/runners"]});return a.jsx(D,{store:t,children:a.jsxs(a.Fragment,{children:[a.jsx(me,{router:n}),a.jsx(v,{})]})})}const xe=Array.from({length:80},(r,t)=>{const n=(t+1).toString().padStart(3,"0"),o=t%2===0;return{workerId:`worker-batch-${n}`,status:o?"busy":"idle",heartbeatStats:o?{queueDepth:t%5+1}:{queueDepth:0},lastHeartbeatTime:e-(t*5+15),lastDequeueTime:e-(t*20+60),lastBusyTime:o?e-(t*10+30):null,lastIdleTime:o?null:e-(t*10+45),currentRolloutId:o?`ro-many-${n}`:null,currentAttemptId:o?`at-many-${n}`:null}}),u={render:()=>Ie(),parameters:{msw:{handlers:k}}},d={render:()=>y(),parameters:{msw:{handlers:k}},play:async({canvasElement:r})=>{const t=b(r);await t.findByText("worker-east");const n=t.getByPlaceholderText("Search by Runner ID");await W.type(n,"worker-west"),await A(()=>{if(t.queryByText("worker-east"))throw new Error("Expected filtered table to hide worker-east");if(!t.queryByText("worker-west"))throw new Error("Expected worker-west to remain visible")})}},m={render:()=>y(),parameters:{msw:{handlers:k}},play:async({canvasElement:r})=>{const t=b(r);await t.findByText("worker-east");const n=await t.findAllByRole("button",{name:/detail/i});await W.click(n[0]);const o=b(document.body);await A(()=>{if(!o.queryByTestId("json-editor-container"))throw new Error("Expected worker detail drawer with JSON view")})}},p={render:()=>y(),parameters:{msw:{handlers:R(xe)}}},h={render:()=>y({theme:"dark"}),parameters:{theme:"dark",msw:{handlers:k}}};u.parameters={...u.parameters,docs:{...u.parameters?.docs,source:{originalSource:`{
  render: () => renderWithinAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...u.parameters?.docs?.source}}};d.parameters={...d.parameters,docs:{...d.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('worker-east');
    const searchInput = canvas.getByPlaceholderText('Search by Runner ID');
    await userEvent.type(searchInput, 'worker-west');
    await waitFor(() => {
      if (canvas.queryByText('worker-east')) {
        throw new Error('Expected filtered table to hide worker-east');
      }
      if (!canvas.queryByText('worker-west')) {
        throw new Error('Expected worker-west to remain visible');
      }
    });
  }
}`,...d.parameters?.docs?.source}}};m.parameters={...m.parameters,docs:{...m.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('worker-east');
    const detailsButtons = await canvas.findAllByRole('button', {
      name: /detail/i
    });
    await userEvent.click(detailsButtons[0]);
    const body = within(document.body);
    await waitFor(() => {
      if (!body.queryByTestId('json-editor-container')) {
        throw new Error('Expected worker detail drawer with JSON view');
      }
    });
  }
}`,...m.parameters?.docs?.source}}};p.parameters={...p.parameters,docs:{...p.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createWorkersHandlers(manyWorkers)
    }
  }
}`,...p.parameters?.docs?.source}}};h.parameters={...h.parameters,docs:{...h.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore({
    theme: 'dark'
  }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...h.parameters?.docs?.source}}};export{h as DarkTheme,u as Default,m as DrawerOpen,p as ManyWorkers,d as Search,Ye as default};
