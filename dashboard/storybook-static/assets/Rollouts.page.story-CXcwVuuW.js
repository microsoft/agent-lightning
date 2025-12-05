import{r as i,j as n,b as xe,a as He}from"./iframe-ByZyHG7Z.js";import{S as Ae,a as je,b as u,w as T,d as X,g as c,u as $}from"./modes-Di9-PqFx.js";import{h as I,A as qe,R as Ce,H as b}from"./AppLayout-xqSbn3cQ.js";import{u as Be,f as m,w as Fe,x as De,y as Oe,z as Me,A as Le,B as We,C as _e,D as Pe,E as $e,F as Z,G as ee,H as Je,I as Ne,J as Ue,K as Ve,L as te,t as Ke,v as ae,M as ze,P as oe,c as Qe,a as Ye,i as Ge,e as Xe}from"./hooks-BqGiFhn9.js";import{A as Ze}from"./AppAlertBanner-yk5DN819.js";import{A as se,c as et}from"./AppDrawer.component-BFL3lspV.js";import{R as tt,a as at}from"./RolloutTable.component-C1WClHvj.js";import{s as rt}from"./selectors-5GhMe3AS.js";import{S as ot}from"./Stack-BYgy4ZNY.js";import{T as st}from"./Title-Czq4HMLJ.js";import{T as nt}from"./TextInput-Ba5dtmf_.js";import{I as lt}from"./IconSearch-Dks1vVM2.js";import"./preload-helper-PPVm8Dsz.js";import"./format-BWW3-KEh.js";import"./table-BuDuxKCV.js";import"./createReactComponent-Cbgg6bZD.js";import"./find-element-ancestor-Cv-4bSct.js";import"./TracesTable.component-v8FAYs_j.js";import"./error-eSvF7V3U.js";import"./IconAlertCircle-BKZdSC5f.js";import"./IconFileDescription-BKfQSike.js";import"./IconTimeline-BvsfcN9Y.js";import"./IconInfoCircle-B0aEKl0P.js";function it({rollout:a,columns:t}){const{data:r,isFetching:o,isError:d,refetch:h}=ze({rolloutId:a.rolloutId,limit:100,sortBy:"sequence_id",sortOrder:"desc"});return n.jsx(at,{rollout:a,attempts:r?.items,isFetching:o,isError:d,onRetry:h,columns:t})}function re(a){const{rolloutId:t,input:r,startTime:o,endTime:d,mode:h,resourcesId:f,status:y,config:U,metadata:g,attempt:S}=a;return{rolloutId:t,input:r,startTime:o,endTime:d,mode:h,resourcesId:f,status:y,config:U,metadata:g,attempt:S??null}}function J(){const a=Be(),t=m(rt),r=m(Fe),o=m(De),d=m(Oe),h=m(Me),f=m(Le),y=m(We),U=m(_e),{data:g,isLoading:S,isFetching:V,isError:K,error:ue,refetch:me}=Pe(U,{pollingInterval:t>0?t:void 0}),pe=i.useCallback(s=>{a($e(s))},[a]),we=i.useCallback(s=>{a(Z(s))},[a]),he=i.useCallback(()=>{a(Z([]))},[a]),fe=i.useCallback(s=>{a(ee(s))},[a]),ge=i.useCallback(()=>{a(ee([]))},[a]),Ie=i.useCallback(s=>{a(Je({column:s.columnAccessor,direction:s.direction}))},[a]),be=i.useCallback(s=>{a(Ne(s))},[a]),Te=i.useCallback(s=>{a(Ue(s))},[a]),ye=i.useCallback(()=>{a(Ve())},[a]),Se=i.useCallback(s=>{a(te({type:"rollout-json",rollout:re(s),attempt:s.attempt??null,isNested:s.isNested}))},[a]),Re=i.useCallback(s=>{a(te({type:"rollout-traces",rollout:re(s),attempt:s.attempt??null,isNested:s.isNested}))},[a]),ve=Array.isArray(g?.items)&&g.items.length>0,ke=S&&!ve;return i.useEffect(()=>{if(K){a(Ke({id:"rollouts-fetch",message:"Unable to refresh rollouts. The list below may be out of date until the connection recovers.",tone:"error"}));return}!S&&!V&&a(ae({id:"rollouts-fetch"}))},[a,K,V,S]),i.useEffect(()=>()=>{a(ae({id:"rollouts-fetch"}))},[a]),n.jsxs(ot,{gap:"md",children:[n.jsx(st,{order:1,children:"Rollouts"}),n.jsx(nt,{placeholder:"Search by Rollout ID",value:r,onChange:s=>pe(s.currentTarget.value),leftSection:n.jsx(lt,{size:16}),"data-testid":"rollouts-search-input",w:"100%",style:{maxWidth:360}}),ke?n.jsx(Ae,{height:360,radius:"md"}):n.jsx(tt,{rollouts:g?.items,totalRecords:g?.total??0,isFetching:V,isError:K,error:ue,searchTerm:r,statusFilters:o,modeFilters:d,sort:y,page:h,recordsPerPage:f,onStatusFilterChange:we,onStatusFilterReset:he,onModeFilterChange:fe,onModeFilterReset:ge,onSortStatusChange:Ie,onPageChange:be,onRecordsPerPageChange:Te,onResetFilters:ye,onRefetch:me,onViewRawJson:Se,onViewTraces:Re,renderRowExpansion:({rollout:s,columns:Ee})=>n.jsx(it,{rollout:s,columns:Ee})})]})}J.__docgenInfo={description:"",methods:[],displayName:"RolloutsPage"};const Zt={title:"Pages/RolloutsPage",component:J,parameters:{layout:"fullscreen",chromatic:{modes:je}}},e=xe,z=[{rolloutId:"ro-7fa3b6e2",input:{task:"Summarize report"},status:"running",mode:"train",resourcesId:"rs-100",startTime:e-1200,endTime:null,attempt:{rolloutId:"ro-7fa3b6e2",attemptId:"at-9001",sequenceId:1,status:"running",startTime:e-1200,endTime:null,workerId:"worker-alpha",lastHeartbeatTime:e-30,metadata:{lastHeartbeatAt:e-30}},config:{retries:0},metadata:{owner:"alice"}},{rolloutId:"ro-116eab45",input:{task:"Classify dataset"},status:"succeeded",mode:"val",resourcesId:"rs-101",startTime:e-5400,endTime:e-3600,attempt:{rolloutId:"ro-116eab45",attemptId:"at-9002",sequenceId:2,status:"succeeded",startTime:e-4e3,endTime:e-3600,workerId:"worker-beta",lastHeartbeatTime:e-3600,metadata:{lastHeartbeatAt:e-3600}},config:{retries:1},metadata:{owner:"bob"}},{rolloutId:"ro-9ae77c11",input:{task:"Evaluate prompt variations"},status:"failed",mode:"test",resourcesId:"rs-102",startTime:e-9600,endTime:e-8400,attempt:{rolloutId:"ro-9ae77c11",attemptId:"at-9005",sequenceId:3,status:"failed",startTime:e-8800,endTime:e-8400,workerId:"worker-gamma",lastHeartbeatTime:e-8400,metadata:{lastHeartbeatAt:e-8400}},config:{retries:2},metadata:{owner:"carol"}}],Q={"ro-7fa3b6e2":[{rolloutId:"ro-7fa3b6e2",attemptId:"at-9001",sequenceId:1,status:"running",startTime:e-1200,endTime:null,workerId:"worker-alpha",lastHeartbeatTime:e-30,metadata:{lastHeartbeatAt:e-30}}],"ro-116eab45":[{rolloutId:"ro-116eab45",attemptId:"at-9000",sequenceId:1,status:"failed",startTime:e-5400,endTime:e-5e3,workerId:"worker-beta",lastHeartbeatTime:e-5e3,metadata:{lastHeartbeatAt:e-5e3}},{rolloutId:"ro-116eab45",attemptId:"at-9002",sequenceId:2,status:"succeeded",startTime:e-4e3,endTime:e-3600,workerId:"worker-beta",lastHeartbeatTime:e-3600,metadata:{lastHeartbeatAt:e-3600}}],"ro-9ae77c11":[{rolloutId:"ro-9ae77c11",attemptId:"at-9003",sequenceId:1,status:"preparing",startTime:e-9600,endTime:e-9300,workerId:"worker-gamma",lastHeartbeatTime:e-9300,metadata:{lastHeartbeatAt:e-9300}},{rolloutId:"ro-9ae77c11",attemptId:"at-9004",sequenceId:2,status:"running",startTime:e-9200,endTime:e-8800,workerId:"worker-delta",lastHeartbeatTime:e-8800,metadata:{lastHeartbeatAt:e-8800}},{rolloutId:"ro-9ae77c11",attemptId:"at-9005",sequenceId:3,status:"failed",startTime:e-8800,endTime:e-8400,workerId:"worker-gamma",lastHeartbeatTime:e-8400,metadata:{lastHeartbeatAt:e-8400}}]},Y={"ro-7fa3b6e2:at-9001":[{rolloutId:"ro-7fa3b6e2",attemptId:"at-9001",sequenceId:1,traceId:"tr-7fa3b6e2-1",spanId:"sp-7fa3b6e2-setup",parentId:null,name:"Initialize rollout",status:{status_code:"OK",description:null},attributes:{step:"init",duration_ms:120},startTime:e-1100,endTime:e-1e3,events:[],links:[],context:{},parent:null,resource:{}},{rolloutId:"ro-7fa3b6e2",attemptId:"at-9001",sequenceId:2,traceId:"tr-7fa3b6e2-1",spanId:"sp-7fa3b6e2-run",parentId:"sp-7fa3b6e2-setup",name:"Execute task",status:{status_code:"OK",description:null},attributes:{step:"run",duration_ms:450},startTime:e-950,endTime:e-500,events:[],links:[],context:{},parent:null,resource:{}}],"ro-116eab45:at-9002":[{rolloutId:"ro-116eab45",attemptId:"at-9002",sequenceId:1,traceId:"tr-116eab45-1",spanId:"sp-116eab45-validate",parentId:null,name:"Validate input",status:{status_code:"OK",description:null},attributes:{step:"validate",duration_ms:80},startTime:e-3800,endTime:e-3720,events:[],links:[],context:{},parent:null,resource:{}},{rolloutId:"ro-116eab45",attemptId:"at-9002",sequenceId:2,traceId:"tr-116eab45-1",spanId:"sp-116eab45-execute",parentId:"sp-116eab45-validate",name:"Execute workflow",status:{status_code:"OK",description:null},attributes:{step:"execute",duration_ms:260},startTime:e-3700,endTime:e-3440,events:[],links:[],context:{},parent:null,resource:{}}],"ro-9ae77c11:at-9005":[{rolloutId:"ro-9ae77c11",attemptId:"at-9005",sequenceId:1,traceId:"tr-9ae77c11-1",spanId:"sp-9ae77c11-fetch",parentId:null,name:"Fetch resources",status:{status_code:"OK",description:null},attributes:{step:"fetch",duration_ms:200},startTime:e-8700,endTime:e-8500,events:[],links:[],context:{},parent:null,resource:{}},{rolloutId:"ro-9ae77c11",attemptId:"at-9005",sequenceId:2,traceId:"tr-9ae77c11-1",spanId:"sp-9ae77c11-run",parentId:"sp-9ae77c11-fetch",name:"Run evaluation",status:{status_code:"ERROR",description:"Worker timeout"},attributes:{step:"evaluate",duration_ms:600},startTime:e-8450,endTime:e-7850,events:[],links:[],context:{},parent:null,resource:{}}]},ct=Array.from({length:160},(a,t)=>({rolloutId:"ro-7fa3b6e2",attemptId:"at-9001",sequenceId:t+1,traceId:`tr-overflow-${Math.floor(t/5)}`,spanId:`sp-overflow-${t+1}`,parentId:t===0?null:`sp-overflow-${t}`,name:`Overflow span ${t+1}`,status:{status_code:"OK",description:null},attributes:{step:`overflow-${t+1}`,duration_ms:20+t%5},startTime:e-1200-t*20,endTime:e-1180-t*20,events:[],links:[],context:{},parent:null,resource:{}})),dt={...Y,"ro-7fa3b6e2:at-9001":ct},ut=Array.from({length:200},(a,t)=>({id:t+1,detail:`Log entry ${t+1} ${"x".repeat(32)}`,timestamp:e-t*2})),p={rolloutId:"ro-json-overflow",attemptId:"at-json-overflow",sequenceId:1,status:"succeeded",startTime:e-600,endTime:e-300,workerId:"worker-scroll",lastHeartbeatTime:e-300,metadata:{notes:"Completed with a very large JSON payload"}},ne=p.endTime??p.startTime+1,N={rolloutId:"ro-json-overflow",input:{task:"Render large JSON",payload:ut,summary:"This rollout includes many log lines to test scroll behavior."},status:"succeeded",mode:"train",resourcesId:"rs-json-overflow",startTime:p.startTime,endTime:ne,attempt:p,config:{retries:0,parameters:{max_steps:200,batch:5}},metadata:{owner:"scroll-tester",description:"Synthetic rollout with oversized JSON payload for storybook validation.",tags:Array.from({length:40},(a,t)=>`tag-${t+1}`)}},mt=[{rolloutId:N.rolloutId,attemptId:p.attemptId,sequenceId:1,traceId:"tr-json-overflow",spanId:"sp-json-root",parentId:null,name:"json-overflow-root",status:{status_code:"OK",description:null},attributes:{detail:"root span"},startTime:p.startTime,endTime:ne,events:[],links:[],context:{},parent:null,resource:{}}],pt=[N,...z],wt={...Q,[N.rolloutId]:[p]},ht={...Y,[`${N.rolloutId}:${p.attemptId}`]:mt},ft=[{rolloutId:"ro-long-duration",input:{task:"Long running training"},status:"running",mode:"train",resourcesId:"rs-200",startTime:e-168*3600,endTime:null,attempt:{rolloutId:"ro-long-duration",attemptId:"at-long-001",sequenceId:1,status:"running",startTime:e-168*3600,endTime:null,workerId:"worker-long",lastHeartbeatTime:e-45,metadata:null},config:{retries:0},metadata:{owner:"delta"}}],gt={"ro-long-duration":[{rolloutId:"ro-long-duration",attemptId:"at-long-001",sequenceId:1,status:"running",startTime:e-168*3600,endTime:null,workerId:"worker-long",lastHeartbeatTime:e-45,metadata:null}]},It=[{rolloutId:"ro-stale-heartbeat",input:{task:"Investigate stale worker"},status:"running",mode:"test",resourcesId:"rs-201",startTime:e-6*3600,endTime:null,attempt:{rolloutId:"ro-stale-heartbeat",attemptId:"at-stale-001",sequenceId:1,status:"running",startTime:e-6*3600,endTime:null,workerId:"worker-stale",lastHeartbeatTime:e-72*3600,metadata:null},config:{retries:0},metadata:{owner:"echo"}}],bt={"ro-stale-heartbeat":[{rolloutId:"ro-stale-heartbeat",attemptId:"at-stale-001",sequenceId:1,status:"running",startTime:e-6*3600,endTime:null,workerId:"worker-stale",lastHeartbeatTime:e-72*3600,metadata:null}]},Tt=[{rolloutId:"ro-status-mismatch",input:{task:"Edge case validation"},status:"running",mode:"val",resourcesId:null,startTime:e-3600,endTime:e-1800,attempt:{rolloutId:"ro-status-mismatch",attemptId:"at-mismatch-003",sequenceId:3,status:"failed",startTime:e-4200,endTime:e-1800,workerId:"worker-mismatch",lastHeartbeatTime:e-1700,metadata:null},config:{retries:3},metadata:{owner:"foxtrot"}}],yt={"ro-status-mismatch":[{rolloutId:"ro-status-mismatch",attemptId:"at-mismatch-001",sequenceId:1,status:"preparing",startTime:e-5400,endTime:e-5e3,workerId:"worker-mismatch",lastHeartbeatTime:e-5e3,metadata:null},{rolloutId:"ro-status-mismatch",attemptId:"at-mismatch-002",sequenceId:2,status:"running",startTime:e-5e3,endTime:e-4200,workerId:"worker-mismatch",lastHeartbeatTime:e-4e3,metadata:null},{rolloutId:"ro-status-mismatch",attemptId:"at-mismatch-003",sequenceId:3,status:"failed",startTime:e-4200,endTime:e-1800,workerId:"worker-mismatch",lastHeartbeatTime:e-1700,metadata:null}]},St=`{"prompt":"${"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(12)}"}`,Rt=[{rolloutId:"ro-long-input",input:St,status:"queuing",mode:"test",resourcesId:"rs-300",startTime:e-120,endTime:null,attempt:null,config:{retries:0},metadata:{owner:"golf"}}],vt={"ro-long-input":[]},le=Array.from({length:120},(a,t)=>{const r=t*90,o=`ro-page-${t+1}`;return{rolloutId:o,input:{item:t+1},status:t%3===0?"running":t%3===1?"failed":"succeeded",mode:t%2===0?"train":"test",resourcesId:t%5===0?`rs-${100+t}`:null,startTime:e-r-300,endTime:t%3===0?null:e-r,attempt:{rolloutId:o,attemptId:`at-page-${t+1}`,sequenceId:1,status:t%3===0?"running":t%3===1?"failed":"succeeded",startTime:e-r-300,endTime:t%3===0?null:e-r,workerId:`worker-${t%7+1}`,lastHeartbeatTime:t%3===0?e-r-60:e-r,metadata:null},config:{},metadata:null}}),kt=Object.fromEntries(le.map(a=>[a.rolloutId,a.attempt?[a.attempt]:[]])),Et=[{rolloutId:"ro-auto-expand",input:{task:"Auto expand test"},status:"running",mode:"train",resourcesId:"rs-400",startTime:e-3600,endTime:null,attempt:{rolloutId:"ro-auto-expand",attemptId:"at-expand-003",sequenceId:3,status:"running",startTime:e-3600,endTime:null,workerId:"worker-auto",lastHeartbeatTime:e-30,metadata:null},config:{},metadata:null}],xt={"ro-auto-expand":[{rolloutId:"ro-auto-expand",attemptId:"at-expand-001",sequenceId:1,status:"preparing",startTime:e-5400,endTime:e-5e3,workerId:"worker-auto",lastHeartbeatTime:e-5e3,metadata:null},{rolloutId:"ro-auto-expand",attemptId:"at-expand-002",sequenceId:2,status:"failed",startTime:e-5e3,endTime:e-4200,workerId:"worker-auto",lastHeartbeatTime:e-4200,metadata:null},{rolloutId:"ro-auto-expand",attemptId:"at-expand-003",sequenceId:3,status:"running",startTime:e-3600,endTime:null,workerId:"worker-auto",lastHeartbeatTime:e-30,metadata:null}]};function ie(a,t){return Qe({config:{...Ge,baseUrl:He,autoRefreshMs:0,...t},rollouts:{...Ye,...a},resources:Xe})}function l(a,t){const r=ie(a,t);return n.jsx(oe,{store:r,children:n.jsxs(n.Fragment,{children:[n.jsx(J,{}),n.jsx(Ze,{}),n.jsx(se,{})]})})}function ce(a,t){const r=ie(a,t),o=et([{path:"/",element:n.jsx(qe,{config:{baseUrl:r.getState().config.baseUrl,autoRefreshMs:r.getState().config.autoRefreshMs}}),children:[{path:"/rollouts",element:n.jsx(J,{})}]}],{initialEntries:["/rollouts"]});return n.jsx(oe,{store:r,children:n.jsxs(n.Fragment,{children:[n.jsx(Ce,{router:o}),n.jsx(se,{})]})})}const w=u(z,Q,Y),Ht=u(z,Q,dt),At=u(pt,wt,ht),R={render:()=>l(),parameters:{msw:{handlers:w}}},v={name:"Within AppLayout",render:()=>ce(),parameters:{msw:{handlers:w}}},k={name:"Within AppLayout (Status Filter)",render:()=>ce({statusFilters:["running"]}),parameters:{msw:{handlers:w}},play:async()=>{await T(()=>{const a=document.querySelector('[data-testid="rollouts-table-container"]'),t=document.querySelector(".mantine-AppShell-main");if(!a||!t)throw new Error("Unable to locate rollout table container or AppShell main region");const r=a.getBoundingClientRect(),o=t.getBoundingClientRect();if(r.right>o.right+1)throw new Error("Rollouts table extends beyond the AppShell content area")})}},E={render:()=>l(void 0,{theme:"dark"}),parameters:{theme:"dark",msw:{handlers:w}}},x={render:()=>l(),parameters:{msw:{handlers:[I.get("*/v1/agl/rollouts",()=>b.json({items:[],limit:0,offset:0,total:0})),I.get("*/v1/agl/rollouts/:rolloutId/attempts",()=>b.json({items:[],limit:0,offset:0,total:0}))]}}},H={render:()=>l(),parameters:{msw:{handlers:[I.get("*/v1/agl/rollouts",()=>b.json({detail:"Internal error"},{status:500})),I.get("*/v1/agl/rollouts/:rolloutId/attempts",()=>b.json({items:[],limit:0,offset:0,total:0},{status:200}))]}}},A={render:()=>l(),parameters:{msw:{handlers:[I.get("*/v1/agl/rollouts",async()=>(await X("infinite"),b.json({items:[],limit:0,offset:0,total:0}))),I.get("*/v1/agl/rollouts/:rolloutId/attempts",async()=>(await X("infinite"),b.json({items:[],limit:0,offset:0,total:0})))]}}},j={render:()=>l(),parameters:{msw:{handlers:u(ft,gt,{})}}},q={render:()=>l(),parameters:{msw:{handlers:u(It,bt,{})}}},C={render:()=>l(),parameters:{msw:{handlers:u(Tt,yt,{})}}},B={render:()=>l(),parameters:{msw:{handlers:u(Rt,vt,{})}}},F={render:()=>l({recordsPerPage:20}),parameters:{msw:{handlers:u(le,kt,{})}}},D={render:()=>l(),parameters:{msw:{handlers:u(Et,xt,{})}},play:async({canvasElement:a})=>{const t=c(a);await t.findByText("ro-auto-expand");const o=t.getByText("ro-auto-expand").closest("tr");if(!o)throw new Error("Unable to locate the rollout row for expansion");await $.click(o),await T(async()=>{await t.findByText("at-expand-001")},{timeout:3e3})}},O={render:()=>l(),parameters:{msw:{handlers:w}},play:async({canvasElement:a})=>{const t=c(a);await t.findByText("ro-7fa3b6e2");const r=t.getByPlaceholderText("Search by Rollout ID");await $.type(r,"ro-116eab45"),await T(()=>{if(t.queryByText("ro-7fa3b6e2"))throw new Error("Expected search to filter out non-matching rollouts");if(!t.queryByText("ro-116eab45"))throw new Error("Expected search to keep the matching rollout visible")})}};async function G(a){const t=c(a);await t.findByText("ro-7fa3b6e2");const o=t.getByText("ro-7fa3b6e2").closest("tr");if(!o)throw new Error("Unable to locate rollout row for traces drawer");const f=c(o).getAllByRole("button",{name:"View traces"})[0];return await $.click(f),c(document.body).findByRole("dialog")}async function de(a,t="ro-7fa3b6e2"){const r=c(a);await r.findByText(t);const d=r.getByText(t).closest("tr");if(!d)throw new Error(`Unable to locate rollout row for ${t}`);const y=c(d).getAllByRole("button",{name:"View raw JSON"})[0];return await $.click(y),c(document.body).findByRole("dialog")}const M={render:()=>l(),parameters:{msw:{handlers:w}},play:async({canvasElement:a})=>{const t=await de(a);await T(async()=>{await c(t).findByText("Attempt"),await c(t).findByText(/worker-alpha/)},{timeout:3e3})}},L={name:"Raw JSON Drawer Scrollable",render:()=>l(),parameters:{msw:{handlers:At}},play:async({canvasElement:a})=>{const r=(await de(a,"ro-json-overflow")).querySelector('[data-testid="json-editor-container"]');if(!r)throw new Error("Unable to locate JSON editor container");await T(()=>{const o=r.querySelector(".monaco-scrollable-element");if(!o)throw new Error("Monaco editor not ready yet");if(o.scrollHeight<=o.clientHeight)throw new Error("Expected JSON content to overflow and allow scrolling")})}},W={render:()=>l(),parameters:{msw:{handlers:w}},play:async({canvasElement:a})=>{await G(a)}},_={name:"Traces Drawer Link",render:()=>l(),parameters:{msw:{handlers:w}},play:async({canvasElement:a})=>{const t=await G(a),o=(await c(t).findByText("View full traces")).getAttribute("href");if(!o)throw new Error("Expected traces drawer to render a link to the traces page");if(!o.includes("rolloutId=ro-7fa3b6e2"))throw new Error(`Link href ${o} is missing rolloutId query parameter`);if(!o.includes("attemptId=at-9001"))throw new Error(`Link href ${o} is missing attemptId query parameter`)}},P={name:"Traces Drawer Scrollable Table",render:()=>l(),parameters:{msw:{handlers:Ht}},play:async({canvasElement:a})=>{const r=(await G(a)).querySelector('[data-testid="traces-drawer-table-container"]');if(!r)throw new Error("Unable to locate traces table container inside drawer");const o=window.getComputedStyle(r).overflowY;if(o!=="auto"&&o!=="scroll")throw new Error("Expected traces table container to allow vertical scrolling");await T(()=>{if(r.scrollHeight<=r.clientHeight)throw new Error("Expected traces table content to overflow and enable scrolling")})}};R.parameters={...R.parameters,docs:{...R.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...R.parameters?.docs?.source}}};v.parameters={...v.parameters,docs:{...v.parameters?.docs,source:{originalSource:`{
  name: 'Within AppLayout',
  render: () => renderWithAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...v.parameters?.docs?.source}}};k.parameters={...k.parameters,docs:{...k.parameters?.docs,source:{originalSource:`{
  name: 'Within AppLayout (Status Filter)',
  render: () => renderWithAppLayout({
    statusFilters: ['running']
  }),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async () => {
    await waitFor(() => {
      const container = document.querySelector<HTMLElement>('[data-testid="rollouts-table-container"]');
      const main = document.querySelector<HTMLElement>('.mantine-AppShell-main');
      if (!container || !main) {
        throw new Error('Unable to locate rollout table container or AppShell main region');
      }
      const containerRect = container.getBoundingClientRect();
      const mainRect = main.getBoundingClientRect();
      if (containerRect.right > mainRect.right + 1) {
        throw new Error('Rollouts table extends beyond the AppShell content area');
      }
    });
  }
}`,...k.parameters?.docs?.source}}};E.parameters={...E.parameters,docs:{...E.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(undefined, {
    theme: 'dark'
  }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...E.parameters?.docs?.source}}};x.parameters={...x.parameters,docs:{...x.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/rollouts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      })), http.get('*/v1/agl/rollouts/:rolloutId/attempts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }))]
    }
  }
}`,...x.parameters?.docs?.source}}};H.parameters={...H.parameters,docs:{...H.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/rollouts', () => HttpResponse.json({
        detail: 'Internal error'
      }, {
        status: 500
      })), http.get('*/v1/agl/rollouts/:rolloutId/attempts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }, {
        status: 200
      }))]
    }
  }
}`,...H.parameters?.docs?.source}}};A.parameters={...A.parameters,docs:{...A.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/rollouts', async () => {
        await delay('infinite');
        return HttpResponse.json({
          items: [],
          limit: 0,
          offset: 0,
          total: 0
        });
      }), http.get('*/v1/agl/rollouts/:rolloutId/attempts', async () => {
        await delay('infinite');
        return HttpResponse.json({
          items: [],
          limit: 0,
          offset: 0,
          total: 0
        });
      })]
    }
  }
}`,...A.parameters?.docs?.source}}};j.parameters={...j.parameters,docs:{...j.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(longDurationRollouts, longDurationAttempts, {})
    }
  }
}`,...j.parameters?.docs?.source}}};q.parameters={...q.parameters,docs:{...q.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(staleHeartbeatRollouts, staleHeartbeatAttempts, {})
    }
  }
}`,...q.parameters?.docs?.source}}};C.parameters={...C.parameters,docs:{...C.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(statusMismatchRollouts, statusMismatchAttempts, {})
    }
  }
}`,...C.parameters?.docs?.source}}};B.parameters={...B.parameters,docs:{...B.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(longInputRollouts, longInputAttempts, {})
    }
  }
}`,...B.parameters?.docs?.source}}};F.parameters={...F.parameters,docs:{...F.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore({
    recordsPerPage: 20
  }),
  parameters: {
    msw: {
      handlers: createMockHandlers(paginationRollouts, paginationAttempts, {})
    }
  }
}`,...F.parameters?.docs?.source}}};D.parameters={...D.parameters,docs:{...D.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(autoExpandRollouts, autoExpandAttempts, {})
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('ro-auto-expand');
    const rolloutCell = canvas.getByText('ro-auto-expand');
    const rolloutRow = rolloutCell.closest('tr');
    if (!rolloutRow) {
      throw new Error('Unable to locate the rollout row for expansion');
    }
    await userEvent.click(rolloutRow);
    await waitFor(async () => {
      await canvas.findByText('at-expand-001');
    }, {
      timeout: 3_000
    });
  }
}`,...D.parameters?.docs?.source}}};O.parameters={...O.parameters,docs:{...O.parameters?.docs,source:{originalSource:`{
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
    await canvas.findByText('ro-7fa3b6e2');
    const searchInput = canvas.getByPlaceholderText('Search by Rollout ID');
    await userEvent.type(searchInput, 'ro-116eab45');
    await waitFor(() => {
      if (canvas.queryByText('ro-7fa3b6e2')) {
        throw new Error('Expected search to filter out non-matching rollouts');
      }
      if (!canvas.queryByText('ro-116eab45')) {
        throw new Error('Expected search to keep the matching rollout visible');
      }
    });
  }
}`,...O.parameters?.docs?.source}}};M.parameters={...M.parameters,docs:{...M.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const drawer = await openRawJsonDrawer(canvasElement);
    await waitFor(async () => {
      await within(drawer).findByText('Attempt');
      await within(drawer).findByText(/worker-alpha/);
    }, {
      timeout: 3_000
    });
  }
}`,...M.parameters?.docs?.source}}};L.parameters={...L.parameters,docs:{...L.parameters?.docs,source:{originalSource:`{
  name: 'Raw JSON Drawer Scrollable',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: jsonOverflowHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const drawer = await openRawJsonDrawer(canvasElement, 'ro-json-overflow');
    const editorContainer = drawer.querySelector('[data-testid="json-editor-container"]') as HTMLElement | null;
    if (!editorContainer) {
      throw new Error('Unable to locate JSON editor container');
    }
    await waitFor(() => {
      const scrollable = editorContainer.querySelector('.monaco-scrollable-element') as HTMLElement | null;
      if (!scrollable) {
        throw new Error('Monaco editor not ready yet');
      }
      if (scrollable.scrollHeight <= scrollable.clientHeight) {
        throw new Error('Expected JSON content to overflow and allow scrolling');
      }
    });
  }
}`,...L.parameters?.docs?.source}}};W.parameters={...W.parameters,docs:{...W.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    await openSampleTracesDrawer(canvasElement);
  }
}`,...W.parameters?.docs?.source}}};_.parameters={..._.parameters,docs:{..._.parameters?.docs,source:{originalSource:`{
  name: 'Traces Drawer Link',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const drawer = await openSampleTracesDrawer(canvasElement);
    const link = await within(drawer).findByText('View full traces');
    const href = link.getAttribute('href');
    if (!href) {
      throw new Error('Expected traces drawer to render a link to the traces page');
    }
    if (!href.includes('rolloutId=ro-7fa3b6e2')) {
      throw new Error(\`Link href \${href} is missing rolloutId query parameter\`);
    }
    if (!href.includes('attemptId=at-9001')) {
      throw new Error(\`Link href \${href} is missing attemptId query parameter\`);
    }
  }
}`,..._.parameters?.docs?.source}}};P.parameters={...P.parameters,docs:{...P.parameters?.docs,source:{originalSource:`{
  name: 'Traces Drawer Scrollable Table',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: overflowHandlers
    }
  },
  play: async ({
    canvasElement
  }) => {
    const drawer = await openSampleTracesDrawer(canvasElement);
    const container = drawer.querySelector('[data-testid="traces-drawer-table-container"]') as HTMLElement | null;
    if (!container) {
      throw new Error('Unable to locate traces table container inside drawer');
    }
    const overflowStyle = window.getComputedStyle(container).overflowY;
    if (overflowStyle !== 'auto' && overflowStyle !== 'scroll') {
      throw new Error('Expected traces table container to allow vertical scrolling');
    }
    await waitFor(() => {
      if (container.scrollHeight <= container.clientHeight) {
        throw new Error('Expected traces table content to overflow and enable scrolling');
      }
    });
  }
}`,...P.parameters?.docs?.source}}};export{D as AutoExpandedAttempt,E as DarkTheme,R as Default,x as EmptyState,A as Loading,j as LongDuration,B as LongInput,F as Pagination,M as RawJsonDrawer,L as RawJsonDrawerScrollable,O as Search,H as ServerError,q as StaleHeartbeat,C as StatusMismatch,W as TracesDrawer,_ as TracesDrawerLink,P as TracesDrawerScrollableTable,v as WithSidebarLayout,k as WithSidebarStatusFilter,Zt as default};
