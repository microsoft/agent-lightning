import{r as o,j as e,B as Q,T as n,G as m,c as p,A as g,D as X,d as v}from"./iframe-ByZyHG7Z.js";import{u as Z,a as ee,g as re,b as te,W as ne,C as N,I as P,d as _,e as U}from"./table-BuDuxKCV.js";import{g as ae}from"./error-eSvF7V3U.js";import{t as h,g as W,b as se}from"./format-BWW3-KEh.js";import{I as ie}from"./IconAlertCircle-BKZdSC5f.js";import{c as ue}from"./createReactComponent-Cbgg6bZD.js";import{I as le}from"./IconFileDescription-BKfQSike.js";import{S as D}from"./Stack-BYgy4ZNY.js";/**
 * @license @tabler/icons-react v3.35.0 - MIT
 *
 * This source code is licensed under the MIT license.
 * See the LICENSE file in the root directory of this source tree.
 */const oe=[["path",{d:"M3 17h4v4h-4z",key:"svg-0"}],["path",{d:"M17 3h4v4h-4z",key:"svg-1"}],["path",{d:"M11 19h5.5a3.5 3.5 0 0 0 0 -7h-8a3.5 3.5 0 0 1 0 -7h4.5",key:"svg-2"}]],de=ue("outline","route-square","RouteSquare",oe),me=[50,100,200,500],ce={name:{minWidth:12.5,priority:0},sequenceId:{fixedWidth:6,priority:1},spanId:{fixedWidth:14,priority:1},traceId:{fixedWidth:24,priority:3},parentId:{fixedWidth:12,priority:2},statusCode:{fixedWidth:8,priority:2},attributeKeys:{minWidth:12.5,priority:2},startTime:{fixedWidth:15,priority:1},endTime:{fixedWidth:15,priority:1},duration:{fixedWidth:10,priority:3},actionsPlaceholder:{fixedWidth:6,priority:0}},ye={UNSET:"gray",OK:"teal",ERROR:"red"};function pe(a){const s=a.status.status_code,d=Object.keys(a.attributes??{}).join(", ")||"",i=h(a.startTime),r=h(a.endTime),t=r&&i?r-i:0;return{...a,statusCode:s,attributeKeys:d,duration:t,actionsPlaceholder:null}}function ge({onShowRollout:a,onShowSpanDetail:s,onParentIdClick:d,spanIds:i}){return[{accessor:"name",title:"Name",sortable:!0,render:({name:r})=>e.jsx(n,{size:"sm",fw:500,children:r})},{accessor:"sequenceId",title:"Seq.",sortable:!0,render:({sequenceId:r})=>e.jsx(n,{size:"sm",children:r})},{accessor:"traceId",title:"Trace ID",sortable:!0,render:({traceId:r})=>e.jsxs(m,{gap:2,children:[e.jsx(n,{size:"sm",children:r}),e.jsx(N,{value:r,children:({copied:t,copy:u})=>e.jsx(p,{label:t?"Copied":"Copy",withArrow:!0,children:e.jsx(g,{"aria-label":`Copy trace ID ${r}`,variant:"subtle",color:t?"teal":"gray",size:"sm",onClick:l=>{l.stopPropagation(),u()},children:t?e.jsx(P,{size:14}):e.jsx(_,{size:14})})})})]})},{accessor:"spanId",title:"Span ID",sortable:!0,render:({spanId:r})=>e.jsxs(m,{gap:2,children:[e.jsx(n,{size:"sm",children:r}),e.jsx(N,{value:r,children:({copied:t,copy:u})=>e.jsx(p,{label:t?"Copied":"Copy",withArrow:!0,children:e.jsx(g,{"aria-label":`Copy span ID ${r}`,variant:"subtle",color:t?"teal":"gray",size:"sm",onClick:l=>{l.stopPropagation(),u()},children:t?e.jsx(P,{size:14}):e.jsx(_,{size:14})})})})]})},{accessor:"parentId",title:"Parent ID",sortable:!0,render:({parentId:r})=>{if(!r)return e.jsx(n,{size:"sm",c:"dimmed",children:"—"});const t=i.has(r),u=t&&typeof d=="function";return e.jsxs(m,{gap:2,children:[e.jsx(n,{size:"sm",c:t?void 0:"red",style:{cursor:u?"pointer":void 0},onClick:l=>{u&&(l.stopPropagation(),d?.(r))},children:r.slice(0,8)}),!t&&e.jsx(p,{label:"Parent span not found in table",withArrow:!0,children:e.jsx(ie,{size:14,color:"red"})})]})}},{accessor:"statusCode",title:"Status",sortable:!0,render:({statusCode:r})=>e.jsx(X,{size:"sm",variant:"light",color:ye[r]??"gray",children:r})},{accessor:"attributeKeys",title:"Attribute Keys",render:({attributeKeys:r})=>r?e.jsx(n,{size:"sm",lineClamp:1,children:r}):e.jsx(n,{size:"sm",c:"dimmed",children:"—"})},{accessor:"startTime",title:"Start Time",sortable:!0,textAlign:"left",render:({startTime:r})=>e.jsx(n,{size:"sm",children:W(h(r))})},{accessor:"endTime",title:"End Time",sortable:!0,textAlign:"left",render:({endTime:r})=>e.jsx(n,{size:"sm",children:W(h(r))})},{accessor:"duration",title:"Duration",sortable:!0,textAlign:"left",render:({duration:r})=>e.jsx(n,{size:"sm",children:se(r)})},{accessor:"actionsPlaceholder",title:"Actions",render:r=>e.jsxs(m,{gap:2,children:[e.jsx(p,{label:"Show rollout",withArrow:!0,disabled:!a,children:e.jsx(g,{"aria-label":"Show rollout",variant:"subtle",color:"gray",onClick:t=>{t.stopPropagation(),a?.(r)},children:e.jsx(de,{size:16})})}),e.jsx(p,{label:"Show span detail",withArrow:!0,disabled:!s,children:e.jsx(g,{"aria-label":"Show span detail",variant:"subtle",color:"gray",onClick:t=>{t.stopPropagation(),s?.(r)},children:e.jsx(le,{size:16})})})]})}]}function ve({spans:a,totalRecords:s,isFetching:d,isError:i,error:r,selectionMessage:t,searchTerm:u,sort:l,page:b,recordsPerPage:q,onSortStatusChange:f,onPageChange:k,onRecordsPerPageChange:M,onResetFilters:R,onRefetch:I,onShowRollout:w,onShowSpanDetail:j,onParentIdClick:S,recordsPerPageOptions:B=me}){const{ref:F,width:C}=Z(),{width:E}=ee(),c=o.useMemo(()=>a?a.map(y=>pe(y)):[],[a]),O=o.useMemo(()=>new Set(c.map(y=>y.spanId)),[c]),z=o.useMemo(()=>ge({onShowRollout:w,onShowSpanDetail:j,onParentIdClick:S,spanIds:O}),[w,j,S,O]),K=o.useMemo(()=>re(C,E),[C,E]),L=o.useMemo(()=>te(z,K,ce),[z,K]),T=o.useMemo(()=>Math.max(1,Math.ceil(Math.max(0,s)/Math.max(1,q))),[q,s]);o.useEffect(()=>{b>T&&k(T)},[k,b,T]);const x=u.trim().length>0,$={columnAccessor:l.column,direction:l.direction},G=o.useCallback(y=>{f(y)},[f]),A=i?ae(r):null,V=i?`Traces are temporarily unavailable${A?` (${A})`:""}.`:"Traces are temporarily unavailable.",H=t?e.jsxs(D,{gap:"sm",align:"center",py:"xl",children:[e.jsx(n,{fw:600,size:"sm",children:t}),e.jsx(n,{size:"sm",c:"dimmed",ta:"center",children:"Choose a rollout and attempt from the controls above to load trace results."})]}):null,Y=e.jsx(D,{gap:"sm",align:"center",py:"lg",children:i?e.jsxs(e.Fragment,{children:[e.jsx(n,{fw:600,size:"sm",children:V}),e.jsx(n,{size:"sm",c:"dimmed",ta:"center",children:"Use the retry button to try again, or adjust the filters to broaden the results."}),e.jsxs(m,{gap:"xs",children:[e.jsx(v,{size:"xs",variant:"light",color:"gray",leftSection:e.jsx(U,{size:14}),onClick:I,children:"Retry"}),x?e.jsx(v,{size:"xs",variant:"subtle",onClick:R,children:"Clear filters"}):null]})]}):e.jsxs(e.Fragment,{children:[e.jsx(n,{fw:600,size:"sm",children:"No traces found"}),e.jsx(n,{size:"sm",c:"dimmed",ta:"center",children:x?"Try adjusting the search to see more results.":"Try refreshing to fetch the latest traces."}),e.jsxs(m,{gap:"xs",children:[e.jsx(v,{size:"xs",variant:"light",leftSection:e.jsx(U,{size:14}),onClick:I,children:"Refresh"}),x?e.jsx(v,{size:"xs",variant:"subtle",onClick:R,children:"Clear filters"}):null]})]})}),J=H??Y;return e.jsx(Q,{ref:F,children:e.jsx(ne,{classNames:{root:"traces-table"},withTableBorder:!0,withColumnBorders:!0,highlightOnHover:!0,verticalAlign:"center",minHeight:c.length===0?500:void 0,idAccessor:"spanId",records:c,columns:L,totalRecords:s,recordsPerPage:q,page:b,onPageChange:k,onRecordsPerPageChange:M,recordsPerPageOptions:B,sortStatus:$,onSortStatusChange:G,fetching:d,loaderSize:"sm",emptyState:c.length===0?J:void 0})})}ve.__docgenInfo={description:"",methods:[],displayName:"TracesTable",props:{spans:{required:!0,tsType:{name:"union",raw:"Span[] | undefined",elements:[{name:"Array",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  status: { status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null };
  attributes: Record<string, any>;
  startTime: Timestamp;
  endTime: Timestamp;

  // The fields below are less frequently used
  events: any;
  links: any;
  context: any;
  parent: any;
  resource: any;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"traceId",value:{name:"string",required:!0}},{key:"spanId",value:{name:"string",required:!0}},{key:"parentId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"name",value:{name:"string",required:!0}},{key:"status",value:{name:"signature",type:"object",raw:"{ status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null }",signature:{properties:[{key:"status_code",value:{name:"union",raw:"'UNSET' | 'OK' | 'ERROR'",elements:[{name:"literal",value:"'UNSET'"},{name:"literal",value:"'OK'"},{name:"literal",value:"'ERROR'"}],required:!0}},{key:"description",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}}]},required:!0}},{key:"attributes",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"number",required:!0}},{key:"events",value:{name:"any",required:!0}},{key:"links",value:{name:"any",required:!0}},{key:"context",value:{name:"any",required:!0}},{key:"parent",value:{name:"any",required:!0}},{key:"resource",value:{name:"any",required:!0}}]}}],raw:"Span[]"},{name:"undefined"}]},description:""},totalRecords:{required:!0,tsType:{name:"number"},description:""},isFetching:{required:!0,tsType:{name:"boolean"},description:""},isError:{required:!0,tsType:{name:"boolean"},description:""},error:{required:!0,tsType:{name:"unknown"},description:""},selectionMessage:{required:!1,tsType:{name:"string"},description:""},searchTerm:{required:!0,tsType:{name:"string"},description:""},sort:{required:!0,tsType:{name:"signature",type:"object",raw:"{ column: string; direction: 'asc' | 'desc' }",signature:{properties:[{key:"column",value:{name:"string",required:!0}},{key:"direction",value:{name:"union",raw:"'asc' | 'desc'",elements:[{name:"literal",value:"'asc'"},{name:"literal",value:"'desc'"}],required:!0}}]}},description:""},page:{required:!0,tsType:{name:"number"},description:""},recordsPerPage:{required:!0,tsType:{name:"number"},description:""},onSortStatusChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(status: DataTableSortStatus<TracesTableRecord>) => void",signature:{arguments:[{type:{name:"DataTableSortStatus",elements:[{name:"intersection",raw:`Span & {
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  status: { status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null };
  attributes: Record<string, any>;
  startTime: Timestamp;
  endTime: Timestamp;

  // The fields below are less frequently used
  events: any;
  links: any;
  context: any;
  parent: any;
  resource: any;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"traceId",value:{name:"string",required:!0}},{key:"spanId",value:{name:"string",required:!0}},{key:"parentId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"name",value:{name:"string",required:!0}},{key:"status",value:{name:"signature",type:"object",raw:"{ status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null }",signature:{properties:[{key:"status_code",value:{name:"union",raw:"'UNSET' | 'OK' | 'ERROR'",elements:[{name:"literal",value:"'UNSET'"},{name:"literal",value:"'OK'"},{name:"literal",value:"'ERROR'"}],required:!0}},{key:"description",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}}]},required:!0}},{key:"attributes",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"number",required:!0}},{key:"events",value:{name:"any",required:!0}},{key:"links",value:{name:"any",required:!0}},{key:"context",value:{name:"any",required:!0}},{key:"parent",value:{name:"any",required:!0}},{key:"resource",value:{name:"any",required:!0}}]}},{name:"signature",type:"object",raw:`{
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"statusCode",value:{name:"string",required:!0}},{key:"attributeKeys",value:{name:"string",required:!0}},{key:"duration",value:{name:"number",required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]}],raw:"DataTableSortStatus<TracesTableRecord>"},name:"status"}],return:{name:"void"}}},description:""},onPageChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(page: number) => void",signature:{arguments:[{type:{name:"number"},name:"page"}],return:{name:"void"}}},description:""},onRecordsPerPageChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(value: number) => void",signature:{arguments:[{type:{name:"number"},name:"value"}],return:{name:"void"}}},description:""},onResetFilters:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onRefetch:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onShowRollout:{required:!1,tsType:{name:"signature",type:"function",raw:"(record: TracesTableRecord) => void",signature:{arguments:[{type:{name:"intersection",raw:`Span & {
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  status: { status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null };
  attributes: Record<string, any>;
  startTime: Timestamp;
  endTime: Timestamp;

  // The fields below are less frequently used
  events: any;
  links: any;
  context: any;
  parent: any;
  resource: any;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"traceId",value:{name:"string",required:!0}},{key:"spanId",value:{name:"string",required:!0}},{key:"parentId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"name",value:{name:"string",required:!0}},{key:"status",value:{name:"signature",type:"object",raw:"{ status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null }",signature:{properties:[{key:"status_code",value:{name:"union",raw:"'UNSET' | 'OK' | 'ERROR'",elements:[{name:"literal",value:"'UNSET'"},{name:"literal",value:"'OK'"},{name:"literal",value:"'ERROR'"}],required:!0}},{key:"description",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}}]},required:!0}},{key:"attributes",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"number",required:!0}},{key:"events",value:{name:"any",required:!0}},{key:"links",value:{name:"any",required:!0}},{key:"context",value:{name:"any",required:!0}},{key:"parent",value:{name:"any",required:!0}},{key:"resource",value:{name:"any",required:!0}}]}},{name:"signature",type:"object",raw:`{
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"statusCode",value:{name:"string",required:!0}},{key:"attributeKeys",value:{name:"string",required:!0}},{key:"duration",value:{name:"number",required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]},name:"record"}],return:{name:"void"}}},description:""},onShowSpanDetail:{required:!1,tsType:{name:"signature",type:"function",raw:"(record: TracesTableRecord) => void",signature:{arguments:[{type:{name:"intersection",raw:`Span & {
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  status: { status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null };
  attributes: Record<string, any>;
  startTime: Timestamp;
  endTime: Timestamp;

  // The fields below are less frequently used
  events: any;
  links: any;
  context: any;
  parent: any;
  resource: any;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"traceId",value:{name:"string",required:!0}},{key:"spanId",value:{name:"string",required:!0}},{key:"parentId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"name",value:{name:"string",required:!0}},{key:"status",value:{name:"signature",type:"object",raw:"{ status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null }",signature:{properties:[{key:"status_code",value:{name:"union",raw:"'UNSET' | 'OK' | 'ERROR'",elements:[{name:"literal",value:"'UNSET'"},{name:"literal",value:"'OK'"},{name:"literal",value:"'ERROR'"}],required:!0}},{key:"description",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}}]},required:!0}},{key:"attributes",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"number",required:!0}},{key:"events",value:{name:"any",required:!0}},{key:"links",value:{name:"any",required:!0}},{key:"context",value:{name:"any",required:!0}},{key:"parent",value:{name:"any",required:!0}},{key:"resource",value:{name:"any",required:!0}}]}},{name:"signature",type:"object",raw:`{
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"statusCode",value:{name:"string",required:!0}},{key:"attributeKeys",value:{name:"string",required:!0}},{key:"duration",value:{name:"number",required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]},name:"record"}],return:{name:"void"}}},description:""},onParentIdClick:{required:!1,tsType:{name:"signature",type:"function",raw:"(parentId: string) => void",signature:{arguments:[{type:{name:"string"},name:"parentId"}],return:{name:"void"}}},description:""},recordsPerPageOptions:{required:!1,tsType:{name:"Array",elements:[{name:"number"}],raw:"number[]"},description:"",defaultValue:{value:"[50, 100, 200, 500]",computed:!1}}}};export{de as I,ve as T,pe as b};
