import{l as We,f as U,u as K,e as ce,i as Le,j as e,B as pe,m as ie,C as On,n as Nn,o as Dn,h as _n,r as T,I as Fe,p as Mn,q as Hn,s as En,t as Vn,v as Bn,w as Wn,x as He,y as Ln,z as _,O as Fn,G as M,T as b,c as re,A as le,D as $e,d as O,E as $n}from"./iframe-ByZyHG7Z.js";import{u as Gn,a as Un,g as Kn,b as Jn,W as Ge,C as Ee,I as Ve,d as Be,e as se}from"./table-BuDuxKCV.js";import{s as Ue,t as me,c as Ke,a as xe,f as Yn,b as Qn,d as Xn}from"./format-BWW3-KEh.js";import{c as Zn}from"./createReactComponent-Cbgg6bZD.js";import{S as $}from"./Stack-BYgy4ZNY.js";import{I as et}from"./IconFileDescription-BKfQSike.js";import{I as nt}from"./IconTimeline-BvsfcN9Y.js";import{I as tt}from"./IconAlertCircle-BKZdSC5f.js";const[at,Se]=We(),[rt,lt]=We();var ge={root:"m_7cda1cd6","root--default":"m_44da308b","root--contrast":"m_e3a01f8",label:"m_1e0e6180",remove:"m_ae386778",group:"m_1dcfd90b"};const ut=Le((t,{gap:a},{size:l})=>({group:{"--pg-gap":a!==void 0?ie(a):ie(l,"pg-gap")}})),je=U((t,a)=>{const l=K("PillGroup",null,t),{classNames:o,className:m,style:c,styles:u,unstyled:i,vars:g,size:v,disabled:n,attributes:r,...p}=l,q=Se()?.size||v||void 0,f=ce({name:"PillGroup",classes:ge,props:l,className:m,style:c,classNames:o,styles:u,unstyled:i,attributes:r,vars:g,varsResolver:ut,stylesCtx:{size:q},rootSelector:"group"});return e.jsx(rt,{value:{size:q,disabled:n},children:e.jsx(pe,{ref:a,size:q,...f("group"),...p})})});je.classes=ge;je.displayName="@mantine/core/PillGroup";const it={variant:"default"},st=Le((t,{radius:a},{size:l})=>({root:{"--pill-fz":ie(l,"pill-fz"),"--pill-height":ie(l,"pill-height"),"--pill-radius":a===void 0?void 0:Nn(a)}})),G=U((t,a)=>{const l=K("Pill",it,t),{classNames:o,className:m,style:c,styles:u,unstyled:i,vars:g,variant:v,children:n,withRemoveButton:r,onRemove:p,removeButtonProps:d,radius:q,size:f,disabled:h,mod:A,attributes:z,...j}=l,R=lt(),H=Se(),E=f||R?.size||void 0,N=H?.variant==="filled"?"contrast":v||"default",D=ce({name:"Pill",classes:ge,props:l,className:m,style:c,classNames:o,styles:u,unstyled:i,attributes:z,vars:g,varsResolver:st,stylesCtx:{size:E}});return e.jsxs(pe,{component:"span",ref:a,variant:N,size:E,...D("root",{variant:N}),mod:[{"with-remove":r&&!h,disabled:h||R?.disabled},A],...j,children:[e.jsx("span",{...D("label"),children:n}),r&&e.jsx(On,{variant:"transparent",radius:q,tabIndex:-1,"aria-hidden":!0,unstyled:i,...d,...D("remove",{className:d?.className,style:d?.style}),onMouseDown:x=>{x.preventDefault(),x.stopPropagation(),d?.onMouseDown?.(x)},onClick:x=>{x.stopPropagation(),p?.(),d?.onClick?.(x)}})]})});G.classes=ge;G.displayName="@mantine/core/Pill";G.Group=je;var Je={field:"m_45c4369d"};const mt={type:"visible"},Ce=U((t,a)=>{const l=K("PillsInputField",mt,t),{classNames:o,className:m,style:c,styles:u,unstyled:i,vars:g,type:v,disabled:n,id:r,pointer:p,mod:d,attributes:q,...f}=l,h=Se(),A=Dn(),z=ce({name:"PillsInputField",classes:Je,props:l,className:m,style:c,classNames:o,styles:u,unstyled:i,attributes:q,rootSelector:"field"}),j=n||h?.disabled;return e.jsx(pe,{component:"input",ref:_n(a,h?.fieldRef),"data-type":v,disabled:j,mod:[{disabled:j,pointer:p},d],...z("field"),...f,id:A?.inputId||r,"aria-invalid":h?.hasError,"aria-describedby":A?.describedBy,type:"text",onMouseDown:R=>!p&&R.stopPropagation()})});Ce.classes=Je;Ce.displayName="@mantine/core/PillsInputField";const ot={size:"sm"},oe=U((t,a)=>{const l=K("PillsInput",ot,t),{children:o,onMouseDown:m,onClick:c,size:u,disabled:i,__staticSelector:g,error:v,variant:n,...r}=l,p=T.useRef(null);return e.jsx(at,{value:{fieldRef:p,size:u,disabled:i,hasError:!!v,variant:n},children:e.jsx(Fe,{size:u,error:v,variant:n,component:"div",ref:a,"data-no-overflow":!0,onMouseDown:d=>{d.preventDefault(),m?.(d),p.current?.focus()},onClick:d=>{d.preventDefault(),d.currentTarget.closest("fieldset")?.disabled||(p.current?.focus(),c?.(d))},...r,multiline:!0,disabled:i,__staticSelector:g||"PillsInput",withAria:!1,children:o})})});oe.displayName="@mantine/core/PillsInput";oe.Field=Ce;function dt({data:t,value:a}){const l=a.map(m=>m.trim().toLowerCase());return t.reduce((m,c)=>(Mn(c)?m.push({group:c.group,items:c.items.filter(u=>l.indexOf(u.value.toLowerCase().trim())===-1)}):l.indexOf(c.value.toLowerCase().trim())===-1&&m.push(c),m),[])}const ct={maxValues:1/0,withCheckIcon:!0,checkIconPosition:"left",hiddenInputValuesDivider:",",clearSearchOnChange:!0,size:"sm"},de=U((t,a)=>{const l=K("MultiSelect",ct,t),{classNames:o,className:m,style:c,styles:u,unstyled:i,vars:g,size:v,value:n,defaultValue:r,onChange:p,onKeyDown:d,variant:q,data:f,dropdownOpened:h,defaultDropdownOpened:A,onDropdownOpen:z,onDropdownClose:j,selectFirstOptionOnChange:R,onOptionSubmit:H,comboboxProps:E,filter:N,limit:D,withScrollArea:x,maxDropdownHeight:ve,searchValue:J,defaultSearchValue:Y,onSearchChange:Q,readOnly:I,disabled:S,onFocus:X,onBlur:V,radius:B,rightSection:ye,rightSectionWidth:be,rightSectionPointerEvents:Z,rightSectionProps:qe,leftSection:w,leftSectionWidth:W,leftSectionPointerEvents:L,leftSectionProps:we,inputContainer:ee,inputWrapperOrder:Ye,withAsterisk:Qe,labelProps:Xe,descriptionProps:Ze,errorProps:en,wrapperProps:nn,description:tn,label:Te,error:Pe,maxValues:an,searchable:C,nothingFoundMessage:Ae,withCheckIcon:rn,checkIconPosition:ln,hidePickedOptions:un,withErrorStyles:sn,name:mn,form:on,id:dn,clearable:cn,clearButtonProps:pn,hiddenInputProps:gn,placeholder:ze,hiddenInputValuesDivider:vn,required:yn,mod:bn,renderOption:qn,onRemove:ke,onClear:wn,scrollAreaProps:Tn,chevronColor:kn,attributes:ne,clearSearchOnChange:fn,...Oe}=l,fe=Hn(dn),he=En(f),P=Vn(he),Ne=T.useRef({}),k=Bn({opened:h,defaultOpened:A,onDropdownOpen:z,onDropdownClose:()=>{j?.(),k.resetSelectedOption()}}),{styleProps:hn,rest:{type:St,autoComplete:In,...Rn}}=Wn(Oe),[y,F]=He({value:n,defaultValue:r,finalValue:[],onChange:p}),[te,xn]=He({value:J,defaultValue:Y,finalValue:"",onChange:Q}),ae=s=>{xn(s),k.resetSelectedOption()},Ie=ce({name:"MultiSelect",classes:{},props:l,classNames:o,styles:u,unstyled:i,attributes:ne}),{resolvedClassNames:De,resolvedStyles:_e}=Ln({props:l,styles:u,classNames:o}),Sn=s=>{d?.(s),s.key===" "&&!C&&(s.preventDefault(),k.toggleDropdown()),s.key==="Backspace"&&te.length===0&&y.length>0&&(ke?.(y[y.length-1]),F(y.slice(0,y.length-1)))},jn=y.map((s,Re)=>{const An=P[s]||Ne.current[s];return e.jsx(G,{withRemoveButton:!I&&!P[s]?.disabled,onRemove:()=>{F(y.filter(zn=>s!==zn)),ke?.(s)},unstyled:i,disabled:S,...Ie("pill"),children:An?.label||s},`${s}-${Re}`)});T.useEffect(()=>{R&&k.selectFirstOption()},[R,te]),T.useEffect(()=>{y.forEach(s=>{s in P&&(Ne.current[s]=P[s])})},[P,y]);const Cn=e.jsx(_.ClearButton,{...pn,onClear:()=>{wn?.(),F([]),ae("")}}),Pn=dt({data:he,value:y}),Me=cn&&y.length>0&&!S&&!I;return e.jsxs(e.Fragment,{children:[e.jsxs(_,{store:k,classNames:De,styles:_e,unstyled:i,size:v,readOnly:I,__staticSelector:"MultiSelect",attributes:ne,onOptionSubmit:s=>{H?.(s),fn&&ae(""),k.updateSelectedOptionIndex("selected"),y.includes(P[s].value)?(F(y.filter(Re=>Re!==P[s].value)),ke?.(P[s].value)):y.length<an&&F([...y,P[s].value])},...E,children:[e.jsx(_.DropdownTarget,{children:e.jsx(oe,{...hn,__staticSelector:"MultiSelect",classNames:De,styles:_e,unstyled:i,size:v,className:m,style:c,variant:q,disabled:S,radius:B,__defaultRightSection:e.jsx(_.Chevron,{size:v,error:Pe,unstyled:i,color:kn}),__clearSection:Cn,__clearable:Me,rightSection:ye,rightSectionPointerEvents:Z||"none",rightSectionWidth:be,rightSectionProps:qe,leftSection:w,leftSectionWidth:W,leftSectionPointerEvents:L,leftSectionProps:we,inputContainer:ee,inputWrapperOrder:Ye,withAsterisk:Qe,labelProps:Xe,descriptionProps:Ze,errorProps:en,wrapperProps:nn,description:tn,label:Te,error:Pe,withErrorStyles:sn,__stylesApiProps:{...l,rightSectionPointerEvents:Z||(Me?"all":"none"),multiline:!0},pointer:!C,onClick:()=>C?k.openDropdown():k.toggleDropdown(),"data-expanded":k.dropdownOpened||void 0,id:fe,required:yn,mod:bn,attributes:ne,children:e.jsxs(G.Group,{attributes:ne,disabled:S,unstyled:i,...Ie("pillsList"),children:[jn,e.jsx(_.EventsTarget,{autoComplete:In,children:e.jsx(oe.Field,{...Rn,ref:a,id:fe,placeholder:ze,type:!C&&!ze?"hidden":"visible",...Ie("inputField"),unstyled:i,onFocus:s=>{X?.(s),C&&k.openDropdown()},onBlur:s=>{V?.(s),k.closeDropdown(),ae("")},onKeyDown:Sn,value:te,onChange:s=>{ae(s.currentTarget.value),C&&k.openDropdown(),R&&k.selectFirstOption()},disabled:S,readOnly:I||!C,pointer:!C})})]})})}),e.jsx(Fn,{data:un?Pn:he,hidden:I||S,filter:N,search:te,limit:D,hiddenWhenEmpty:!Ae,withScrollArea:x,maxDropdownHeight:ve,filterOptions:C,value:y,checkIconPosition:ln,withCheckIcon:rn,nothingFoundMessage:Ae,unstyled:i,labelId:Te?`${fe}-label`:void 0,"aria-label":Te?void 0:Oe["aria-label"],renderOption:qn,scrollAreaProps:Tn})]}),e.jsx(_.HiddenInput,{name:mn,valuesDivider:vn,value:y,form:on,disabled:S,...gn})]})});de.classes={...Fe.classes,..._.classes};de.displayName="@mantine/core/MultiSelect";/**
 * @license @tabler/icons-react v3.35.0 - MIT
 *
 * This source code is licensed under the MIT license.
 * See the LICENSE file in the root directory of this source tree.
 */const pt=[["path",{d:"M19.933 13.041a8 8 0 1 1 -9.925 -8.788c3.899 -1 7.935 1.007 9.425 4.747",key:"svg-0"}],["path",{d:"M20 4v5h-5",key:"svg-1"}]],gt=Zn("outline","reload","Reload",pt),vt=["queuing","preparing","running","failed","succeeded","cancelled","requeuing"],yt={failed:"red",preparing:"violet",running:"blue",succeeded:"teal",timeout:"orange",unresponsive:"orange"},bt={cancelled:"gray",failed:"red",preparing:"violet",queuing:"gray",requeuing:"gray",running:"blue",succeeded:"teal"},qt=["train","val","test"],wt=[50,100,200,500],Tt={rolloutId:{fixedWidth:12.5,priority:0},actionsPlaceholder:{fixedWidth:6.5,priority:0},inputText:{minWidth:14,priority:1},statusValue:{fixedWidth:10,priority:1},startTimestamp:{fixedWidth:12,priority:2},durationSeconds:{fixedWidth:10,priority:2},attemptId:{fixedWidth:12,priority:3},resourcesId:{fixedWidth:10,priority:3},mode:{fixedWidth:8,priority:3},lastHeartbeatTimestamp:{fixedWidth:10,priority:3},workerId:{fixedWidth:10,priority:3}};function kt(t){return!t||t.lastHeartbeatTime==null||Number.isNaN(t.lastHeartbeatTime)?null:t.lastHeartbeatTime}function ft(t){const a=t.attempt,l=t.input===null||typeof t.input>"u"?"—":typeof t.input=="string"?t.input:Ue(t.input),o=me(a?.startTime??t.startTime),m=me(a?.endTime??t.endTime),c=Ke(o,m),u=a?.status,i=a?.sequenceId,g=u&&u!==t.status?`${t.status}-${u}`:t.status;return{...t,attempt:a??null,attemptId:a?.attemptId??null,attemptSequence:a?.sequenceId??null,isNested:!1,canExpand:!!(i&&i>1),inputText:l,attemptStatus:u,statusValue:g,startTimestamp:o,durationSeconds:c,lastHeartbeatTimestamp:t.attempt?.lastHeartbeatTime??null,workerId:a?.workerId??null,actionsPlaceholder:null}}function ht(t,a){const l=t.input===null||typeof t.input>"u"?"—":typeof t.input=="string"?t.input:Ue(t.input),o=me(a.startTime??t.startTime),m=me(a.endTime),c=Ke(o,m),u=kt(a);return{...t,attempt:a,attemptId:a.attemptId,attemptSequence:a.sequenceId,isNested:!0,canExpand:!1,inputText:l,attemptStatus:a.status,statusValue:a.status,startTimestamp:o,durationSeconds:c,lastHeartbeatTimestamp:u,workerId:a.workerId??null,actionsPlaceholder:null}}function ue(t,a){const l=a==="rollout"?bt[t]??"gray":yt[t]??"gray";return e.jsx($e,{size:"sm",variant:"light",color:l,children:xe(t)})}function It({statusFilters:t,onStatusFilterChange:a,onStatusFilterReset:l,modeFilters:o,onModeFilterChange:m,onModeFilterReset:c,onViewRawJson:u,onViewTraces:i}){const g=vt.map(n=>({value:n,label:xe(n)})),v=qt.map(n=>({value:n,label:xe(n)}));return[{accessor:"rolloutId",title:"Rollout",sortable:!0,render:({rolloutId:n})=>e.jsxs(M,{gap:2,children:[e.jsx(b,{fw:500,size:"sm",children:n}),e.jsx(Ee,{value:n,children:({copied:r,copy:p})=>e.jsx(re,{label:r?"Copied":"Copy",withArrow:!0,children:e.jsx(le,{"aria-label":`Copy rollout ID ${n}`,variant:"subtle",color:r?"teal":"gray",size:"sm",onClick:d=>{d.stopPropagation(),p()},children:r?e.jsx(Ve,{size:14}):e.jsx(Be,{size:14})})})})]})},{accessor:"attemptId",title:"Attempt",sortable:!0,render:({attemptId:n,attemptSequence:r,isNested:p})=>e.jsxs(M,{gap:2,children:[e.jsx(b,{size:"sm",c:n?void 0:"dimmed",children:n??"—"}),n&&e.jsx(Ee,{value:n,children:({copied:d,copy:q})=>e.jsx(re,{label:d?"Copied":"Copy",withArrow:!0,children:e.jsx(le,{"aria-label":`Copy attempt ID ${n}`,variant:"subtle",color:d?"teal":"gray",size:"sm",onClick:f=>{f.stopPropagation(),q()},children:d?e.jsx(Ve,{size:14}):e.jsx(Be,{size:14})})})}),r&&(p||r>1)&&e.jsx($e,{leftSection:e.jsx(gt,{size:12}),pl:6,pr:6,children:r})]})},{accessor:"inputText",title:"Input",render:({inputText:n})=>e.jsx(b,{size:"sm",ff:"monospace",c:"dimmed",lineClamp:1,title:n,style:{width:"100%",wordBreak:"break-all",overflow:"hidden"},children:n})},{accessor:"statusValue",title:"Status",sortable:!0,filter:({close:n})=>e.jsxs($,{gap:"xs",children:[e.jsx(de,{label:"Status",description:"Filter rollouts by status",data:g,value:t,placeholder:"Select statuses...",searchable:!0,clearable:!0,comboboxProps:{withinPortal:!1},onChange:r=>a(r)}),e.jsx(O,{variant:"light",size:"xs",onClick:()=>{l(),n()},disabled:t.length===0,children:"Clear"})]}),filtering:t.length>0,render:({status:n,attemptStatus:r,isNested:p})=>p?e.jsx(M,{gap:4,children:ue(r??"unknown","attempt")}):r&&r!==n?e.jsxs(M,{gap:4,children:[ue(n,"rollout"),e.jsx(b,{size:"sm",c:"dimmed",children:"—"}),ue(r,"attempt")]}):ue(n,"rollout")},{accessor:"resourcesId",title:"Resources",sortable:!0,render:({resourcesId:n})=>e.jsx(b,{size:"sm",c:n?void 0:"dimmed",children:n??"—"})},{accessor:"mode",title:"Mode",sortable:!0,filter:({close:n})=>e.jsxs($,{gap:"xs",children:[e.jsx(de,{label:"Mode",description:"Filter rollouts by mode",data:v,value:o,placeholder:"Select modes...",searchable:!0,clearable:!0,comboboxProps:{withinPortal:!1},onChange:r=>m(r)}),e.jsx(O,{variant:"light",size:"xs",onClick:()=>{c(),n()},disabled:o.length===0,children:"Clear"})]}),filtering:o.length>0,render:({mode:n})=>e.jsx(b,{size:"sm",c:n?void 0:"dimmed",children:n??"—"})},{accessor:"startTimestamp",title:"Start Time",sortable:!0,textAlign:"left",render:({startTimestamp:n})=>e.jsx(b,{size:"sm",children:Yn(n)})},{accessor:"durationSeconds",title:"Duration",sortable:!0,textAlign:"left",render:({durationSeconds:n})=>e.jsx(b,{size:"sm",children:Qn(n)})},{accessor:"lastHeartbeatTimestamp",title:"Last Heartbeat",sortable:!0,textAlign:"left",render:({lastHeartbeatTimestamp:n,attempt:r,isNested:p})=>!r&&p?e.jsx(b,{size:"sm",c:"dimmed",children:"—"}):e.jsx(b,{size:"sm",children:Xn(n)})},{accessor:"workerId",title:"Worker",sortable:!0,render:({workerId:n})=>e.jsx(b,{size:"sm",c:n?void 0:"dimmed",children:n??"—"})},{accessor:"actionsPlaceholder",title:"Actions",render:n=>e.jsxs(M,{gap:4,children:[e.jsx(re,{label:"View raw JSON",withArrow:!0,disabled:!u,children:e.jsx(le,{"aria-label":"View raw JSON",variant:"subtle",color:"gray",onClick:r=>{r.stopPropagation(),u?.(n)},children:e.jsx(et,{size:16})})}),e.jsx(re,{label:"View traces",withArrow:!0,disabled:!i,children:e.jsx(le,{"aria-label":"View traces",variant:"subtle",color:"gray",onClick:r=>{r.stopPropagation(),i?.(n)},children:e.jsx(nt,{size:16})})})]})}]}function Rt({rollouts:t,totalRecords:a,isFetching:l,isError:o,error:m,searchTerm:c,statusFilters:u,modeFilters:i,sort:g,page:v,recordsPerPage:n,onStatusFilterChange:r,onStatusFilterReset:p,onModeFilterChange:d,onModeFilterReset:q,onSortStatusChange:f,onPageChange:h,onRecordsPerPageChange:A,onResetFilters:z,onRefetch:j,onViewRawJson:R,onViewTraces:H,recordsPerPageOptions:E=wt,renderRowExpansion:N}){const[D,x]=T.useState([]),{ref:ve,width:J}=Gn(),{width:Y}=Un(),Q=T.useMemo(()=>Kn(J,Y),[J,Y]),I=T.useMemo(()=>t?t.map(w=>ft(w)):[],[t]),S=T.useMemo(()=>It({statusFilters:u,onStatusFilterChange:r,onStatusFilterReset:p,modeFilters:i,onModeFilterChange:d,onModeFilterReset:q,onViewRawJson:R,onViewTraces:H}),[u,r,p,i,d,q,R,H]),X=T.useMemo(()=>Jn(S,Q,Tt),[S,Q]),V=T.useMemo(()=>Math.max(1,Math.ceil(Math.max(0,a)/Math.max(1,n))),[n,a]);T.useEffect(()=>{v>V&&h(V)},[h,v,V]),T.useEffect(()=>{x(w=>w.filter(W=>I.some(L=>L.rolloutId===W&&L.canExpand)))},[I]);const B=c.trim().length>0||u.length>0||i.length>0,ye={columnAccessor:g.column,direction:g.direction},be=T.useCallback(w=>{f(w)},[f]),Z=o&&m&&typeof m=="object"&&"status"in m?`Rollouts are temporarily unavailable (status: ${String(m.status)}).`:"Rollouts are temporarily unavailable.",qe=e.jsx($,{gap:"sm",align:"center",py:"lg",children:o?e.jsxs(e.Fragment,{children:[e.jsx(b,{fw:600,size:"sm",children:Z}),e.jsx(b,{size:"sm",c:"dimmed",ta:"center",children:"Use the retry button to try again, or adjust the filters to broaden the results."}),e.jsxs(M,{gap:"xs",children:[e.jsx(O,{size:"xs",variant:"light",color:"gray",leftSection:e.jsx(se,{size:14}),onClick:j,children:"Retry"}),B?e.jsx(O,{size:"xs",variant:"subtle",onClick:z,children:"Clear filters"}):null]})]}):e.jsxs(e.Fragment,{children:[e.jsx(b,{fw:600,size:"sm",children:"No rollouts found"}),e.jsx(b,{size:"sm",c:"dimmed",ta:"center",children:B?"Try adjusting the search or filters to see more results.":"Try refreshing to fetch the latest rollouts."}),e.jsxs(M,{gap:"xs",children:[e.jsx(O,{size:"xs",variant:"light",leftSection:e.jsx(se,{size:14}),onClick:j,children:"Refresh"}),B?e.jsx(O,{size:"xs",variant:"subtle",onClick:z,children:"Clear filters"}):null]})]})});return e.jsx(pe,{ref:ve,"data-testid":"rollouts-table-container",children:e.jsx(Ge,{classNames:{root:"rollouts-table"},withTableBorder:!0,withColumnBorders:!0,highlightOnHover:!0,verticalAlign:"center",minHeight:I.length===0?500:void 0,idAccessor:"rolloutId",records:I,columns:X,totalRecords:a,recordsPerPage:n,page:v,onPageChange:h,onRecordsPerPageChange:A,recordsPerPageOptions:E,sortStatus:ye,onSortStatusChange:be,fetching:l,loaderSize:"sm",emptyState:I.length===0?qe:void 0,rowExpansion:N?{allowMultiple:!0,expandable:({record:w})=>w.canExpand,expanded:{recordIds:D,onRecordIdsChange:w=>{x(W=>(typeof w=="function"?w(W):w??[]).map(String).filter(we=>I.some(ee=>ee.rolloutId===we&&ee.canExpand)))}},content:({record:w})=>N({rollout:w,columns:X})}:void 0})})}function xt({rollout:t,attempts:a,isFetching:l,isError:o,onRetry:m,columns:c}){const u=T.useMemo(()=>a?a.map(g=>ht(t,g)).sort((g,v)=>(v.attemptSequence??0)-(g.attemptSequence??0)).filter(g=>g.attemptSequence!==t.attempt?.sequenceId):[],[a,t]);if(o&&!u.length)return e.jsx($n,{color:"red",variant:"light",icon:e.jsx(tt,{size:16}),children:e.jsxs($,{gap:"xs",children:[e.jsx(b,{size:"sm",children:"Unable to load attempts for this rollout."}),e.jsx(O,{size:"xs",variant:"light",leftSection:e.jsx(se,{size:14}),onClick:m,children:"Retry"})]})});const i=e.jsxs($,{gap:"xs",align:"center",py:"md",children:[e.jsx(b,{size:"sm",c:"dimmed",children:"No attempts found for this rollout."}),e.jsx(O,{size:"xs",variant:"light",leftSection:e.jsx(se,{size:14}),onClick:m,children:"Refresh"})]});return e.jsx(Ge,{classNames:{root:"rollouts-table rollouts-table--nested"},withColumnBorders:!0,noHeader:!0,minHeight:0,idAccessor:"attemptId",verticalAlign:"center",fetching:l,loaderSize:"sm",records:u,columns:c,emptyState:u.length===0?i:void 0})}Rt.__docgenInfo={description:"",methods:[],displayName:"RolloutTable",props:{rollouts:{required:!0,tsType:{name:"union",raw:"Rollout[] | undefined",elements:[{name:"Array",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}}],raw:"Rollout[]"},{name:"undefined"}]},description:""},totalRecords:{required:!0,tsType:{name:"number"},description:""},isFetching:{required:!0,tsType:{name:"boolean"},description:""},isError:{required:!0,tsType:{name:"boolean"},description:""},error:{required:!0,tsType:{name:"unknown"},description:""},searchTerm:{required:!0,tsType:{name:"string"},description:""},statusFilters:{required:!0,tsType:{name:"Array",elements:[{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}]}],raw:"RolloutStatus[]"},description:""},modeFilters:{required:!0,tsType:{name:"Array",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]}],raw:"RolloutMode[]"},description:""},sort:{required:!0,tsType:{name:"signature",type:"object",raw:`{
  column: string;
  direction: SortDirection;
}`,signature:{properties:[{key:"column",value:{name:"string",required:!0}},{key:"direction",value:{name:"union",raw:"'asc' | 'desc'",elements:[{name:"literal",value:"'asc'"},{name:"literal",value:"'desc'"}],required:!0}}]}},description:""},page:{required:!0,tsType:{name:"number"},description:""},recordsPerPage:{required:!0,tsType:{name:"number"},description:""},onStatusFilterChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(values: RolloutStatus[]) => void",signature:{arguments:[{type:{name:"Array",elements:[{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}]}],raw:"RolloutStatus[]"},name:"values"}],return:{name:"void"}}},description:""},onStatusFilterReset:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onModeFilterChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(values: RolloutMode[]) => void",signature:{arguments:[{type:{name:"Array",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]}],raw:"RolloutMode[]"},name:"values"}],return:{name:"void"}}},description:""},onModeFilterReset:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onSortStatusChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(status: DataTableSortStatus<RolloutTableRecord>) => void",signature:{arguments:[{type:{name:"DataTableSortStatus",elements:[{name:"intersection",raw:`Rollout & {
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}},{name:"signature",type:"object",raw:`{
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"attemptId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"attemptSequence",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"isNested",value:{name:"boolean",required:!0}},{key:"canExpand",value:{name:"boolean",required:!0}},{key:"inputText",value:{name:"string",required:!0}},{key:"attemptStatus",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"statusValue",value:{name:"string",required:!0}},{key:"startTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"durationSeconds",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"lastHeartbeatTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]}],raw:"DataTableSortStatus<RolloutTableRecord>"},name:"status"}],return:{name:"void"}}},description:""},onPageChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(page: number) => void",signature:{arguments:[{type:{name:"number"},name:"page"}],return:{name:"void"}}},description:""},onRecordsPerPageChange:{required:!0,tsType:{name:"signature",type:"function",raw:"(value: number) => void",signature:{arguments:[{type:{name:"number"},name:"value"}],return:{name:"void"}}},description:""},onResetFilters:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onRefetch:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},onViewRawJson:{required:!1,tsType:{name:"signature",type:"function",raw:"(record: RolloutTableRecord) => void",signature:{arguments:[{type:{name:"intersection",raw:`Rollout & {
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}},{name:"signature",type:"object",raw:`{
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"attemptId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"attemptSequence",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"isNested",value:{name:"boolean",required:!0}},{key:"canExpand",value:{name:"boolean",required:!0}},{key:"inputText",value:{name:"string",required:!0}},{key:"attemptStatus",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"statusValue",value:{name:"string",required:!0}},{key:"startTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"durationSeconds",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"lastHeartbeatTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]},name:"record"}],return:{name:"void"}}},description:""},onViewTraces:{required:!1,tsType:{name:"signature",type:"function",raw:"(record: RolloutTableRecord) => void",signature:{arguments:[{type:{name:"intersection",raw:`Rollout & {
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}},{name:"signature",type:"object",raw:`{
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"attemptId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"attemptSequence",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"isNested",value:{name:"boolean",required:!0}},{key:"canExpand",value:{name:"boolean",required:!0}},{key:"inputText",value:{name:"string",required:!0}},{key:"attemptStatus",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"statusValue",value:{name:"string",required:!0}},{key:"startTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"durationSeconds",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"lastHeartbeatTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]},name:"record"}],return:{name:"void"}}},description:""},recordsPerPageOptions:{required:!1,tsType:{name:"Array",elements:[{name:"number"}],raw:"number[]"},description:"",defaultValue:{value:"[50, 100, 200, 500]",computed:!1}},renderRowExpansion:{required:!1,tsType:{name:"signature",type:"function",raw:`(context: {
  rollout: Rollout;
  columns: DataTableColumn<RolloutTableRecord>[];
}) => ReactNode`,signature:{arguments:[{type:{name:"signature",type:"object",raw:`{
  rollout: Rollout;
  columns: DataTableColumn<RolloutTableRecord>[];
}`,signature:{properties:[{key:"rollout",value:{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]},required:!0}},{key:"columns",value:{name:"Array",elements:[{name:"DataTableColumn",elements:[{name:"intersection",raw:`Rollout & {
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]},required:!0},{name:"signature",type:"object",raw:`{
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"attemptId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"attemptSequence",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"isNested",value:{name:"boolean",required:!0}},{key:"canExpand",value:{name:"boolean",required:!0}},{key:"inputText",value:{name:"string",required:!0}},{key:"attemptStatus",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"statusValue",value:{name:"string",required:!0}},{key:"startTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"durationSeconds",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"lastHeartbeatTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]}],raw:"DataTableColumn<RolloutTableRecord>"}],raw:"DataTableColumn<RolloutTableRecord>[]",required:!0}}]}},name:"context"}],return:{name:"ReactNode"}}},description:""}}};xt.__docgenInfo={description:"",methods:[],displayName:"RolloutAttemptsTable",props:{rollout:{required:!0,tsType:{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}},description:""},attempts:{required:!0,tsType:{name:"union",raw:"Attempt[] | undefined",elements:[{name:"Array",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}}],raw:"Attempt[]"},{name:"undefined"}]},description:""},isFetching:{required:!0,tsType:{name:"boolean"},description:""},isError:{required:!0,tsType:{name:"boolean"},description:""},onRetry:{required:!0,tsType:{name:"signature",type:"function",raw:"() => void",signature:{arguments:[],return:{name:"void"}}},description:""},columns:{required:!0,tsType:{name:"Array",elements:[{name:"DataTableColumn",elements:[{name:"intersection",raw:`Rollout & {
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"input",value:{name:"any",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"mode",value:{name:"union",raw:"RolloutMode | null",elements:[{name:"union",raw:"'train' | 'val' | 'test'",elements:[{name:"literal",value:"'train'"},{name:"literal",value:"'val'"},{name:"literal",value:"'test'"}]},{name:"null"}],required:!0}},{key:"resourcesId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing'",elements:[{name:"literal",value:"'queuing'"},{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'cancelled'"},{name:"literal",value:"'requeuing'"}],required:!0}},{key:"config",value:{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>",required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}},{key:"attempt",value:{name:"union",raw:"Attempt | null",elements:[{name:"signature",type:"object",raw:`{
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
}`,signature:{properties:[{key:"rolloutId",value:{name:"string",required:!0}},{key:"attemptId",value:{name:"string",required:!0}},{key:"sequenceId",value:{name:"number",required:!0}},{key:"startTime",value:{name:"number",required:!0}},{key:"endTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"status",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"lastHeartbeatTime",value:{name:"union",raw:"Timestamp | null",elements:[{name:"number",required:!0},{name:"null"}],required:!0}},{key:"metadata",value:{name:"union",raw:"Record<string, any> | null",elements:[{name:"Record",elements:[{name:"string"},{name:"any"}],raw:"Record<string, any>"},{name:"null"}],required:!0}}]}},{name:"null"}],required:!0}}]}},{name:"signature",type:"object",raw:`{
  attemptId: string | null;
  attemptSequence: number | null;
  isNested: boolean;
  canExpand: boolean;
  inputText: string;
  attemptStatus?: AttemptStatus;
  statusValue: string;
  startTimestamp: number | null;
  durationSeconds: number | null;
  lastHeartbeatTimestamp: number | null;
  workerId: string | null;
  actionsPlaceholder?: null;
}`,signature:{properties:[{key:"attemptId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"attemptSequence",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"isNested",value:{name:"boolean",required:!0}},{key:"canExpand",value:{name:"boolean",required:!0}},{key:"inputText",value:{name:"string",required:!0}},{key:"attemptStatus",value:{name:"union",raw:"'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout'",elements:[{name:"literal",value:"'preparing'"},{name:"literal",value:"'running'"},{name:"literal",value:"'failed'"},{name:"literal",value:"'succeeded'"},{name:"literal",value:"'unresponsive'"},{name:"literal",value:"'timeout'"}],required:!1}},{key:"statusValue",value:{name:"string",required:!0}},{key:"startTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"durationSeconds",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"lastHeartbeatTimestamp",value:{name:"union",raw:"number | null",elements:[{name:"number"},{name:"null"}],required:!0}},{key:"workerId",value:{name:"union",raw:"string | null",elements:[{name:"string"},{name:"null"}],required:!0}},{key:"actionsPlaceholder",value:{name:"null",required:!1}}]}}]}],raw:"DataTableColumn<RolloutTableRecord>"}],raw:"DataTableColumn<RolloutTableRecord>[]"},description:""}}};export{Rt as R,xt as a,ft as b};
