import{r,j as l,G as Ve,H as Ne,d as _t,b as Dt,a as Ot}from"./iframe-ByZyHG7Z.js";import{S as $t,a as Qt,b as B,g as P,w as A,u as M,d as Ee}from"./modes-Di9-PqFx.js";import{h as I,A as Vt,R as Nt,H as v}from"./AppLayout-xqSbn3cQ.js";import{u as Wt,f as b,R as Ut,S as zt,T as Kt,U as Gt,V as Yt,W as Xt,X as Zt,Y as Jt,D as xe,Z as z,M as ea,_ as ta,$ as H,a0 as K,t as aa,v as We,a1 as ra,a2 as Ue,a3 as sa,a4 as na,a5 as oa,a6 as la,L as ze,a7 as ca,P as Ze,c as ia,d as ua,i as da,e as ma,a as pa}from"./hooks-BqGiFhn9.js";import{u as ha,M as ga,A as Je,c as fa}from"./AppDrawer.component-BFL3lspV.js";import{A as wa}from"./AppAlertBanner-yk5DN819.js";import{T as ya}from"./TracesTable.component-v8FAYs_j.js";import{s as Ia}from"./selectors-5GhMe3AS.js";import{g as Ae}from"./error-eSvF7V3U.js";import{a as Ke}from"./format-BWW3-KEh.js";import{S as Ge}from"./Stack-BYgy4ZNY.js";import{T as va}from"./Title-Czq4HMLJ.js";import{T as Ta}from"./TextInput-Ba5dtmf_.js";import{I as Sa}from"./IconSearch-Dks1vVM2.js";import{M as G,I as ba}from"./table-BuDuxKCV.js";import{c as Ra}from"./createReactComponent-Cbgg6bZD.js";import"./preload-helper-PPVm8Dsz.js";import"./IconTimeline-BvsfcN9Y.js";import"./IconAlertCircle-BKZdSC5f.js";import"./IconInfoCircle-B0aEKl0P.js";import"./IconFileDescription-BKfQSike.js";import"./find-element-ancestor-Cv-4bSct.js";function Ea(t,a,n={leading:!1}){const[u,i]=r.useState(t),T=r.useRef(!1),d=r.useRef(null),g=r.useRef(!1),y=r.useCallback(()=>window.clearTimeout(d.current),[]);return r.useEffect(()=>{T.current&&(!g.current&&n.leading?(g.current=!0,i(t)):(y(),d.current=window.setTimeout(()=>{g.current=!1,i(t)},a)))},[t,n.leading,a]),r.useEffect(()=>(T.current=!0,y),[]),[u,y]}/**
 * @license @tabler/icons-react v3.35.0 - MIT
 *
 * This source code is licensed under the MIT license.
 * See the LICENSE file in the root directory of this source tree.
 */const xa=[["path",{d:"M6 9l6 6l6 -6",key:"svg-0"}]],Aa=Ra("outline","chevron-down","ChevronDown",xa),Ye=[{value:"table",label:"Table View",disabled:!1},{value:"waterfall",label:"Waterfall View (Coming Soon)",disabled:!0},{value:"tree",label:"Tree View (Coming Soon)",disabled:!0}];function Ba(t){return t.length?[...t].sort((a,n)=>a.sequenceId-n.sequenceId).at(-1)??null:null}function Be(t,a){if(!a.length)return t;let n=!1;const u={...t};for(const i of a){const T=u[i.rolloutId];(!T||T!==i)&&(u[i.rolloutId]=i,n=!0)}return n?u:t}function he(){const t=Wt(),a=b(Ia),[n,u]=ha(),i=n.toString(),[T,d]=r.useState(null),[g,y]=r.useState(""),[q]=Ea(g,300),w=q.trim(),S=w.length>0,[m,R]=r.useState({}),E=n.has("rolloutId"),_=n.has("attemptId"),[C]=r.useState(E),Ce=E?n.get("rolloutId")||null:void 0,Fe=_?n.get("attemptId")||null:void 0,s=b(Ut),p=b(zt),Le=b(Kt),lt=b(Gt),ct=b(Yt),it=b(Xt),He=b(Zt),ut=b(Jt),dt=r.useMemo(()=>({limit:100,offset:0,sortBy:"start_time",sortOrder:"desc"}),[]),{data:D,isLoading:F,isFetching:O,isError:ge,error:Me}=xe(dt,{pollingInterval:a>0?a:void 0}),x=D?.items??[],mt=r.useMemo(()=>S?{limit:20,offset:0,sortBy:"start_time",sortOrder:"desc",rolloutIdContains:w}:null,[w,S]),{data:pt,isFetching:ht}=xe(mt??z),$=r.useMemo(()=>!s||m[s]?null:{limit:20,offset:0,sortBy:"start_time",sortOrder:"desc",rolloutIdContains:s},[s,m]),{data:Q,isFetching:fe}=xe($??z),V=pt?.items??[],we=Q?.items??[];r.useEffect(()=>{x.length>0&&R(e=>Be(e,x))},[x]),r.useEffect(()=>{V.length>0&&R(e=>Be(e,V))},[V]),r.useEffect(()=>{we.length>0&&R(e=>Be(e,we))},[we]);const L=s?m[s]??null:null,gt=s!==null?{rolloutId:s,limit:200,sortBy:"sequence_id",sortOrder:"desc"}:z,{data:f,isFetching:N,isError:ye,error:qe}=ea(gt,{pollingInterval:a>0?a:void 0}),{data:ft,isFetching:Ie,isError:W,error:ve,refetch:_e}=ta(ut??z,{pollingInterval:a>0?a:void 0}),De=s!==null&&!m[s];r.useEffect(()=>{if(D){if(D.total===0){s!==null&&t(H(null));return}if(s===null){!C&&x[0]&&t(H(x[0].rolloutId));return}if(De&&!(S&&w===s)){if($){if(fe||!Q||F)return;Q.items.length===0&&t(H(null));return}t(H(null))}}},[x,t,C,w,Q,fe,$,s,m,S,D,F,De]),r.useEffect(()=>{if(!s){p!==null&&t(K(null));return}if(f&&f.items.length>0){if(!(p?f.items.some(o=>o.attemptId===p):!1)){const o=Ba(f.items);o&&o.attemptId!==p&&t(K(o.attemptId))}return}if(p===null){const e=L?.attempt?.attemptId??null;e!==p&&t(K(e))}},[f,p,t,s,L]);const Te=S?V:x,Oe=(S?ht:O)||!!($&&fe),$e=r.useMemo(()=>{const e=Te.map(o=>({value:o.rolloutId,label:o.rolloutId}));return s&&!Te.some(o=>o.rolloutId===s)&&e.push({value:s,label:s}),e},[s,Te]),U=r.useMemo(()=>{if(f&&f.items.length>0)return[...f.items].sort((e,o)=>o.sequenceId-e.sequenceId).map(e=>({value:e.attemptId,label:`Attempt ${e.sequenceId} (${e.attemptId}) - ${Ke(e.status)}`}));if(L?.attempt){const e=L.attempt;return[{value:e.attemptId,label:`Attempt ${e.sequenceId} (${e.attemptId}) - ${Ke(e.status)}`}]}return[]},[f,L]),wt=r.useMemo(()=>s?U.length===0?"No Attempt":"Latest Attempt":"Select Attempt",[U.length,s]),Qe=ft,yt=Qe?.items??[],It=Qe?.total??0,vt=[50,100,200,500],Tt=F&&x.length===0,St=Ie||O||N,j=r.useMemo(()=>{if(!s&&!p)return"Select a rollout and attempt to view traces.";if(!s)return"Select a rollout to view traces.";if(!p)return"Select an attempt to view traces."},[p,s]),bt=j?[]:yt,Rt=j?!1:W,Et=j?void 0:ve,xt=j?!1:St;r.useEffect(()=>{if(ge||ye||W){const o=ge&&Ae(Me)||ye&&Ae(qe)||W&&Ae(ve)||null,Se=o?` (${o})`:"";t(aa({id:"traces-fetch",message:`Unable to refresh traces${Se}. The table may be out of date until the connection recovers.`,tone:"error"}));return}!F&&!O&&!Ie&&!N&&t(We({id:"traces-fetch"}))},[qe,N,ye,t,Me,O,ge,F,ve,Ie,W]),r.useEffect(()=>()=>{t(We({id:"traces-fetch"}))},[t]),r.useEffect(()=>{const e={};E&&(e.rolloutId=Ce),_&&(e.attemptId=Fe),Object.keys(e).length>0&&t(ra(e)),d(o=>o===i?o:i)},[Fe,t,_,E,Ce,i]),r.useEffect(()=>{if(T!==i)return;const e=new URLSearchParams(n);let o=!1;s?e.get("rolloutId")!==s&&(e.set("rolloutId",s),o=!0):e.has("rolloutId")&&(e.delete("rolloutId"),o=!0),s&&p?e.get("attemptId")!==p&&(e.set("attemptId",p),o=!0):e.has("attemptId")&&(e.delete("attemptId"),o=!0),o&&u(e,{replace:!0})},[p,T,s,n,i,u]);const At=r.useCallback(e=>{t(Ue(e))},[t]),Bt=r.useCallback(e=>{t(sa({column:e.columnAccessor,direction:e.direction}))},[t]),Pt=r.useCallback(e=>{t(na(e))},[t]),kt=r.useCallback(e=>{t(oa(e))},[t]),jt=r.useCallback(()=>{t(la())},[t]),Ct=r.useCallback(()=>{s&&_e()},[_e,s]),Ft=r.useCallback(e=>{const o=m[e.rolloutId];if(!o)return;const be=(f?.items??[]).find(Re=>Re.attemptId===e.attemptId)??o.attempt??null;t(ze({type:"rollout-json",rollout:o,attempt:be,isNested:!1}))},[f,t,m]),Lt=r.useCallback(e=>{const o=m[e.rolloutId]??null,be=(f?.items??[]).find(Re=>Re.attemptId===e.attemptId)??o?.attempt??null;t(ze({type:"trace-detail",span:e,rollout:o,attempt:be}))},[f,t,m]),Ht=r.useCallback(e=>{t(Ue(e))},[t]),Mt=r.useCallback(e=>{t(ca(e))},[t]),qt=Ye.find(e=>e.value===He)?.label??"Table View";return l.jsxs(Ge,{gap:"md",children:[l.jsxs(Ve,{justify:"space-between",align:"flex-start",children:[l.jsxs(Ge,{gap:"sm",style:{flex:1,minWidth:0},children:[l.jsx(va,{order:1,children:"Traces"}),l.jsxs(Ve,{gap:"md",wrap:"wrap",children:[l.jsx(Ne,{data:$e,value:s??null,onChange:e=>{e!==s&&t(H(e)),y("")},searchable:!0,searchValue:g,onSearchChange:e=>{y(e??"")},placeholder:"Select rollout","aria-label":"Select rollout",nothingFoundMessage:Oe?"Loading...":S?"No matching rollouts":"No rollouts",comboboxProps:{withinPortal:!0},onDropdownOpen:()=>{y("")},onDropdownClose:()=>{y("")},w:260,disabled:$e.length===0&&!Oe}),l.jsx(Ne,{data:U,value:p??null,onChange:e=>{e!==p&&t(K(e))},searchable:!0,placeholder:wt,"aria-label":"Select attempt",nothingFoundMessage:N?"Loading...":"No attempts",comboboxProps:{withinPortal:!0},w:280,disabled:!s||U.length===0}),l.jsx(Ta,{value:Le,onChange:e=>At(e.currentTarget.value),placeholder:"Search spans","aria-label":"Search spans",leftSection:l.jsx(Sa,{size:16}),w:280})]})]}),l.jsxs(G,{shadow:"md",position:"bottom-end",withinPortal:!0,children:[l.jsx(G.Target,{children:l.jsx(_t,{variant:"light",rightSection:l.jsx(Aa,{size:16}),"aria-label":"Change traces view",children:qt})}),l.jsx(G.Dropdown,{children:Ye.map(e=>l.jsx(G.Item,{disabled:e.disabled,leftSection:e.value===He&&!e.disabled?l.jsx(ba,{size:14}):null,onClick:()=>{e.disabled||Mt(e.value)},children:e.label},e.value))})]})]}),l.jsx($t,{visible:Tt,radius:"md",children:l.jsx(ya,{spans:bt,totalRecords:j?0:It,isFetching:xt,isError:Rt,error:Et,selectionMessage:j,searchTerm:Le,sort:it,page:lt,recordsPerPage:ct,onSortStatusChange:Bt,onPageChange:Pt,onRecordsPerPageChange:kt,onResetFilters:jt,onRefetch:Ct,onShowRollout:Ft,onShowSpanDetail:Lt,onParentIdClick:Ht,recordsPerPageOptions:vt})})]})}he.__docgenInfo={description:"",methods:[],displayName:"TracesPage"};const ur={title:"Pages/TracesPage",component:he,parameters:{layout:"fullscreen",chromatic:{modes:Qt}}},c=Dt,Pe=[{rolloutId:"ro-traces-001",input:{task:"Generate onboarding flow"},status:"running",mode:"train",resourcesId:"rs-traces-001",startTime:c-1800,endTime:null,attempt:{rolloutId:"ro-traces-001",attemptId:"at-traces-001",sequenceId:1,status:"running",startTime:c-1800,endTime:null,workerId:"worker-delta",lastHeartbeatTime:c-30,metadata:{region:"us-east-1"}},config:{retries:0},metadata:{owner:"ava"}},{rolloutId:"ro-traces-002",input:{task:"Classify support emails"},status:"succeeded",mode:"val",resourcesId:"rs-traces-002",startTime:c-5400,endTime:c-3600,attempt:{rolloutId:"ro-traces-002",attemptId:"at-traces-004",sequenceId:4,status:"succeeded",startTime:c-4e3,endTime:c-3600,workerId:"worker-epsilon",lastHeartbeatTime:c-3600,metadata:{region:"us-west-2"}},config:{retries:2},metadata:{owner:"ben"}}],ke={"ro-traces-001":[{rolloutId:"ro-traces-001",attemptId:"at-traces-001",sequenceId:1,status:"running",startTime:c-1800,endTime:null,workerId:"worker-delta",lastHeartbeatTime:c-30,metadata:{region:"us-east-1"}},{rolloutId:"ro-traces-001",attemptId:"at-traces-002",sequenceId:2,status:"failed",startTime:c-5400,endTime:c-4800,workerId:"worker-theta",lastHeartbeatTime:c-4800,metadata:{error:"Network timeout"}}],"ro-traces-002":[{rolloutId:"ro-traces-002",attemptId:"at-traces-004",sequenceId:4,status:"succeeded",startTime:c-4e3,endTime:c-3600,workerId:"worker-epsilon",lastHeartbeatTime:c-3600,metadata:{region:"us-west-2"}}]},et={"ro-traces-001:at-traces-001":[{rolloutId:"ro-traces-001",attemptId:"at-traces-001",sequenceId:1,traceId:"tr-001",spanId:"sp-001",parentId:null,name:"Initialize rollout",status:{status_code:"OK",description:null},attributes:{stage:"init",duration_ms:120},startTime:c-1600,endTime:c-1580,events:[],links:[],context:{},parent:null,resource:{}},{rolloutId:"ro-traces-001",attemptId:"at-traces-001",sequenceId:2,traceId:"tr-001",spanId:"sp-002",parentId:"sp-001",name:"Fetch resources",status:{status_code:"OK",description:null},attributes:{endpoint:"/resources/latest",duration_ms:240},startTime:c-1580,endTime:c-1540,events:[],links:[],context:{},parent:null,resource:{}}],"ro-traces-001:at-traces-002":[{rolloutId:"ro-traces-001",attemptId:"at-traces-002",sequenceId:1,traceId:"tr-002",spanId:"sp-101",parentId:null,name:"Initialize rollout",status:{status_code:"ERROR",description:"Timeout"},attributes:{stage:"init",duration_ms:600},startTime:c-5300,endTime:c-4700,events:[],links:[],context:{},parent:null,resource:{}}],"ro-traces-002:at-traces-004":[{rolloutId:"ro-traces-002",attemptId:"at-traces-004",sequenceId:1,traceId:"tr-200",spanId:"sp-201",parentId:null,name:"Load dataset",status:{status_code:"OK",description:null},attributes:{records:1200,duration_ms:420},startTime:c-3800,endTime:c-3720,events:[],links:[],context:{},parent:null,resource:{}},{rolloutId:"ro-traces-002",attemptId:"at-traces-004",sequenceId:2,traceId:"tr-200",spanId:"sp-202",parentId:"sp-201",name:"Classify batch",status:{status_code:"OK",description:null},attributes:{batch:1,duration_ms:320},startTime:c-3720,endTime:c-3660,events:[],links:[],context:{},parent:null,resource:{}}]},Pa={},ka={},pe=Pe[1],ja=[pe],Ca={[pe.rolloutId]:ke[pe.rolloutId]??[]},Fa=Object.fromEntries(Object.entries(et).filter(([t])=>t.startsWith(`${pe.rolloutId}:`))),tt={rolloutId:"ro-traces-no-attempt",input:{task:"Legacy rollout without attempts"},status:"failed",mode:"train",resourcesId:"rs-traces-no-attempt",startTime:c-7200,endTime:c-7e3,attempt:null,config:{retries:0},metadata:{owner:"casey"}},La=[tt],Ha={[tt.rolloutId]:[]},Ma={},Xe=["ava","ben","carla","diego"];function at(t,a){const n={},u={};return{rollouts:Array.from({length:a},(T,d)=>{const g=`ro-${t}-${String(d+1).padStart(3,"0")}`,y=["running","succeeded","failed"],q=["train","val","test"],w=y[d%y.length],S=q[d%q.length],m=c-(d+1)*420,R=w==="running"?null:m+240,E=`${g}-attempt`,C={rolloutId:g,attemptId:E,sequenceId:1,status:w==="failed"?"failed":w==="succeeded"?"succeeded":"running",startTime:m,endTime:R,workerId:`worker-${String.fromCharCode(97+d%26)}`,lastHeartbeatTime:R??m+180,metadata:{region:d%2===0?"us-east-1":"eu-west-1"}};return n[g]=[C],u[`${g}:${E}`]=[{rolloutId:g,attemptId:E,sequenceId:1,traceId:`tr-${t}-${d+1}`,spanId:`sp-${t}-${d+1}-root`,parentId:null,name:`Synthetic root span ${t} ${d+1}`,status:{status_code:w==="failed"?"ERROR":"OK",description:w==="failed"?"Synthetic failure":null},attributes:{"trace.sample":d+1,duration_ms:240},startTime:m,endTime:R??m+240,events:[],links:[],context:{},parent:null,resource:{}}],{rolloutId:g,input:{task:`Synthetic trace ${d+1}`},status:w,mode:S,resourcesId:`rs-${t}-${d%7+1}`,startTime:m,endTime:R,attempt:C,config:{retries:d%3},metadata:{owner:Xe[d%Xe.length]}}}),attemptsByRollout:n,spansByAttempt:u}}const{rollouts:rt,attemptsByRollout:st,spansByAttempt:nt}=at("many",24),{rollouts:qa,attemptsByRollout:_a,spansByAttempt:Da}=at("vast",160);function k(t){return B(Pe,ke,et,t)}function Oa(){return[I.get("*/v1/agl/rollouts",async()=>(await Ee(1200),v.json({detail:"Request timed out"},{status:504,statusText:"Timeout"}))),I.get("*/v1/agl/rollouts/:rolloutId/attempts",async({params:t})=>(await Ee(1200),v.json({detail:"Request timed out",rolloutId:t.rolloutId},{status:504,statusText:"Timeout"}))),I.get("*/v1/agl/spans",async()=>(await Ee(1200),v.json({detail:"Request timed out"},{status:504,statusText:"Timeout"})))]}const $a=B(Pe,ke);function ot(t,a){return ia({config:{...da,baseUrl:Ot,...a},rollouts:pa,resources:ma,traces:{...ua,...t}})}function h(t,a){const n=ot(t,a);return l.jsx(Ze,{store:n,children:l.jsxs(ga,{initialEntries:["/traces"],children:[l.jsx(he,{}),l.jsx(wa,{}),l.jsx(Je,{})]})})}function je(t,a,n="/traces"){const u=ot(t,a),i=fa([{path:"/",element:l.jsx(Vt,{config:{baseUrl:u.getState().config.baseUrl,autoRefreshMs:u.getState().config.autoRefreshMs}}),children:[{path:"/traces",element:l.jsx(he,{})}]}],{initialEntries:[n]});return l.jsx(Ze,{store:u,children:l.jsxs(l.Fragment,{children:[l.jsx(Nt,{router:i}),l.jsx(Je,{})]})})}const Y={render:()=>h(),parameters:{msw:{handlers:k()}}},X={name:"Within AppLayout",render:()=>je(),parameters:{msw:{handlers:k()}}},Z={name:"Loads From Query Params",render:()=>je(void 0,void 0,"/traces?rolloutId=ro-traces-002&attemptId=at-traces-004"),parameters:{msw:{handlers:k()}},play:async({canvasElement:t})=>{const a=P(t),n=await a.findByLabelText("Select rollout");await A(()=>{if(n.value!=="ro-traces-002")throw new Error("Expected rollout select to use value from query string")});const u=await a.findByLabelText("Select attempt");await A(()=>{if(u.value.indexOf("at-traces-004")===-1)throw new Error("Expected attempt select to use value from query string")})}},J={name:"Missing Rollout From Query Params",render:()=>je(void 0,void 0,"/traces?rolloutId=ro-missing-999"),parameters:{msw:{handlers:B(rt,st,nt)}},play:async({canvasElement:t})=>{const a=P(t),n=await a.findByLabelText("Select rollout");await A(()=>{if(n.value!=="")throw new Error("Expected rollout select to remain empty when the query rollout does not exist")}),await A(()=>{if(!a.getByText("Select a rollout and attempt to view traces."))throw new Error("Expected empty selection message when rollout query param is invalid")})}},ee={render:()=>h(void 0,{theme:"dark"}),parameters:{theme:"dark",msw:{handlers:k()}}},te={render:()=>h(),parameters:{msw:{handlers:B([],Pa,ka)}}},ae={render:()=>h(),parameters:{msw:{handlers:B(ja,Ca,Fa)}}},re={render:()=>h(),parameters:{msw:{handlers:B(La,Ha,Ma)}},play:async({canvasElement:t})=>{const n=await P(t).findByLabelText("Select attempt");await A(()=>{if(n.placeholder!=="No Attempt")throw new Error('Expected attempt select placeholder to read "No Attempt" when no attempts are available')})}},se={render:()=>h(),parameters:{msw:{handlers:B(rt,st,nt)}}},ne={render:()=>h(),parameters:{msw:{handlers:B(qa,_a,Da)}},play:async({canvasElement:t})=>{const a=P(t),n=await a.findByLabelText("Select rollout");await M.click(n);for(let i=0;i<12;i++)await M.type(n,"{backspace}");await M.type(n,"ro-vast-150"),await A(()=>{if(!P(document.body).queryByText("ro-vast-150"))throw new Error("Expected remote rollout search to return IDs beyond the initial list")});const u=P(document.body).getByText("ro-vast-150");await M.click(u),await A(()=>{if(n.value!=="ro-vast-150")throw new Error("Expected rollout select to use the searched rollout ID")}),await a.findByText("Synthetic root span vast 150")}},oe={render:()=>h(),parameters:{msw:{handlers:k()}},play:async({canvasElement:t})=>{const a=P(t);await a.findByLabelText("Search spans");const n=a.getByLabelText("Search spans");await M.type(n,"Fetch"),await A(()=>{if(!a.queryByText("Fetch resources"))throw new Error("Expected matching span to be displayed after searching");if(a.queryByText("Initialize rollout"))throw new Error("Expected non-matching spans to be filtered out")})}},le={render:()=>h(),parameters:{msw:{handlers:k(800)}}},ce={render:()=>h(),parameters:{msw:{handlers:Oa()}}},ie={render:()=>h({attemptId:"at-traces-002",rolloutId:"ro-traces-001"}),parameters:{msw:{handlers:k()}}},ue={render:()=>h(),parameters:{msw:{handlers:[I.get("*/v1/agl/rollouts",()=>v.json({items:[],limit:0,offset:0,total:0},{status:200})),I.get("*/v1/agl/rollouts/:rolloutId/attempts",()=>v.json({items:[],limit:0,offset:0,total:0},{status:200})),I.get("*/v1/agl/spans",()=>v.json({detail:"server error"},{status:500}))]}}},de={render:()=>h(),parameters:{msw:{handlers:[I.get("*/v1/agl/rollouts",()=>v.text("not valid json",{status:200,headers:{"Content-Type":"application/json"}})),I.get("*/v1/agl/rollouts/:rolloutId/attempts",()=>v.json({items:[],limit:0,offset:0,total:0})),I.get("*/v1/agl/spans",()=>v.json({items:[],limit:0,offset:0,total:0}))]}}},me={render:()=>h(),parameters:{msw:{handlers:[...$a,I.get("*/v1/agl/spans",()=>v.text("not valid json",{status:200,headers:{"Content-Type":"application/json"}}))]}}};Y.parameters={...Y.parameters,docs:{...Y.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createHandlers()
    }
  }
}`,...Y.parameters?.docs?.source}}};X.parameters={...X.parameters,docs:{...X.parameters?.docs,source:{originalSource:`{
  name: 'Within AppLayout',
  render: () => renderTracesPageWithAppLayout(),
  parameters: {
    msw: {
      handlers: createHandlers()
    }
  }
}`,...X.parameters?.docs?.source}}};Z.parameters={...Z.parameters,docs:{...Z.parameters?.docs,source:{originalSource:`{
  name: 'Loads From Query Params',
  render: () => renderTracesPageWithAppLayout(undefined, undefined, '/traces?rolloutId=ro-traces-002&attemptId=at-traces-004'),
  parameters: {
    msw: {
      handlers: createHandlers()
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    const rolloutInput = (await canvas.findByLabelText('Select rollout')) as HTMLInputElement;
    await waitFor(() => {
      if (rolloutInput.value !== 'ro-traces-002') {
        throw new Error('Expected rollout select to use value from query string');
      }
    });
    const attemptInput = (await canvas.findByLabelText('Select attempt')) as HTMLInputElement;
    await waitFor(() => {
      if (attemptInput.value.indexOf('at-traces-004') === -1) {
        throw new Error('Expected attempt select to use value from query string');
      }
    });
  }
}`,...Z.parameters?.docs?.source}}};J.parameters={...J.parameters,docs:{...J.parameters?.docs,source:{originalSource:`{
  name: 'Missing Rollout From Query Params',
  render: () => renderTracesPageWithAppLayout(undefined, undefined, '/traces?rolloutId=ro-missing-999'),
  parameters: {
    msw: {
      handlers: createMockHandlers(manyRollouts, manyAttemptsByRollout, manySpansByAttempt)
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    const rolloutInput = (await canvas.findByLabelText('Select rollout')) as HTMLInputElement;
    await waitFor(() => {
      if (rolloutInput.value !== '') {
        throw new Error('Expected rollout select to remain empty when the query rollout does not exist');
      }
    });
    await waitFor(() => {
      if (!canvas.getByText('Select a rollout and attempt to view traces.')) {
        throw new Error('Expected empty selection message when rollout query param is invalid');
      }
    });
  }
}`,...J.parameters?.docs?.source}}};ee.parameters={...ee.parameters,docs:{...ee.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(undefined, {
    theme: 'dark'
  }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: createHandlers()
    }
  }
}`,...ee.parameters?.docs?.source}}};te.parameters={...te.parameters,docs:{...te.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers([], emptyAttemptsByRollout, emptySpansByAttempt)
    }
  }
}`,...te.parameters?.docs?.source}}};ae.parameters={...ae.parameters,docs:{...ae.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(singleRollouts, singleAttemptsByRollout, singleSpansByAttempt)
    }
  }
}`,...ae.parameters?.docs?.source}}};re.parameters={...re.parameters,docs:{...re.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(noAttemptRollouts, noAttemptAttemptsByRollout, noAttemptSpansByAttempt)
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    const attemptInput = (await canvas.findByLabelText('Select attempt')) as HTMLInputElement;
    await waitFor(() => {
      if (attemptInput.placeholder !== 'No Attempt') {
        throw new Error('Expected attempt select placeholder to read "No Attempt" when no attempts are available');
      }
    });
  }
}`,...re.parameters?.docs?.source}}};se.parameters={...se.parameters,docs:{...se.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(manyRollouts, manyAttemptsByRollout, manySpansByAttempt)
    }
  }
}`,...se.parameters?.docs?.source}}};ne.parameters={...ne.parameters,docs:{...ne.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(vastRollouts, vastAttemptsByRollout, vastSpansByAttempt)
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    const rolloutTrigger = (await canvas.findByLabelText('Select rollout')) as HTMLInputElement;
    await userEvent.click(rolloutTrigger);
    for (let i = 0; i < 'ro-vast-001'.length + 1; i++) {
      await userEvent.type(rolloutTrigger, '{backspace}');
    }
    await userEvent.type(rolloutTrigger, 'ro-vast-150');
    await waitFor(() => {
      const option = within(document.body).queryByText('ro-vast-150');
      if (!option) {
        throw new Error('Expected remote rollout search to return IDs beyond the initial list');
      }
    });
    const option = within(document.body).getByText('ro-vast-150');
    await userEvent.click(option);
    await waitFor(() => {
      if (rolloutTrigger.value !== 'ro-vast-150') {
        throw new Error('Expected rollout select to use the searched rollout ID');
      }
    });
    await canvas.findByText('Synthetic root span vast 150');
  }
}`,...ne.parameters?.docs?.source}}};oe.parameters={...oe.parameters,docs:{...oe.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createHandlers()
    }
  },
  play: async ({
    canvasElement
  }) => {
    const canvas = within(canvasElement);
    await canvas.findByLabelText('Search spans');
    const searchInput = canvas.getByLabelText('Search spans');
    await userEvent.type(searchInput, 'Fetch');
    await waitFor(() => {
      if (!canvas.queryByText('Fetch resources')) {
        throw new Error('Expected matching span to be displayed after searching');
      }
      if (canvas.queryByText('Initialize rollout')) {
        throw new Error('Expected non-matching spans to be filtered out');
      }
    });
  }
}`,...oe.parameters?.docs?.source}}};le.parameters={...le.parameters,docs:{...le.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createHandlers(800)
    }
  }
}`,...le.parameters?.docs?.source}}};ce.parameters={...ce.parameters,docs:{...ce.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createRequestTimeoutHandlers()
    }
  }
}`,...ce.parameters?.docs?.source}}};ie.parameters={...ie.parameters,docs:{...ie.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage({
    attemptId: 'at-traces-002',
    rolloutId: 'ro-traces-001'
  }),
  parameters: {
    msw: {
      handlers: createHandlers()
    }
  }
}`,...ie.parameters?.docs?.source}}};ue.parameters={...ue.parameters,docs:{...ue.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/rollouts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }, {
        status: 200
      })), http.get('*/v1/agl/rollouts/:rolloutId/attempts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }, {
        status: 200
      })), http.get('*/v1/agl/spans', () => HttpResponse.json({
        detail: 'server error'
      }, {
        status: 500
      }))]
    }
  }
}`,...ue.parameters?.docs?.source}}};de.parameters={...de.parameters,docs:{...de.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/rollouts', () => HttpResponse.text('not valid json', {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        }
      })), http.get('*/v1/agl/rollouts/:rolloutId/attempts', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      })), http.get('*/v1/agl/spans', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }))]
    }
  }
}`,...de.parameters?.docs?.source}}};me.parameters={...me.parameters,docs:{...me.parameters?.docs,source:{originalSource:`{
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [...rolloutsAndAttemptsHandlers, http.get('*/v1/agl/spans', () => HttpResponse.text('not valid json', {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        }
      }))]
    }
  }
}`,...me.parameters?.docs?.source}}};export{ie as AttemptScoped,ee as DarkTheme,Y as DefaultView,te as EmptyState,ne as LargeDatasetSearch,le as LoadingState,se as ManyResults,J as MissingRolloutQuery,re as NoAttemptPlaceholder,me as ParseFailure,Z as QueryParams,ce as RequestTimeout,de as RolloutParseFailure,oe as Search,ue as ServerError,ae as SingleResult,X as WithSidebarLayout,ur as default};
