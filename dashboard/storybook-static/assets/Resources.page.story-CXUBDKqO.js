import{r as o,j as s,b as G,a as J}from"./iframe-ByZyHG7Z.js";import{S as K,a as V,c as R,g as X,u as Z,w as ee,d as re}from"./modes-Di9-PqFx.js";import{h as S,A as se,R as te,H as w}from"./AppLayout-xqSbn3cQ.js";import{u as ae,f as p,s as ne,g as oe,h as ce,j as pe,k as me,l as ie,m as le,n as de,o as ue,p as he,q as ge,t as _e,v as P,P as C,c as fe,i as ye,e as ve,a as xe}from"./hooks-BqGiFhn9.js";import{A as Re}from"./AppAlertBanner-yk5DN819.js";import{A as W,c as Se}from"./AppDrawer.component-BFL3lspV.js";import{R as we}from"./ResourcesTable.component-DCuimg2z.js";import{R as be}from"./ResourcesTree.component-Y1gdellh.js";import{s as Te}from"./selectors-5GhMe3AS.js";import{g as je}from"./error-eSvF7V3U.js";import{S as Ee}from"./Stack-BYgy4ZNY.js";import{T as ke}from"./Title-Czq4HMLJ.js";import{T as Ae}from"./TextInput-Ba5dtmf_.js";import{I as Ie}from"./IconSearch-Dks1vVM2.js";import"./preload-helper-PPVm8Dsz.js";import"./format-BWW3-KEh.js";import"./table-BuDuxKCV.js";import"./createReactComponent-Cbgg6bZD.js";import"./find-element-ancestor-Cv-4bSct.js";import"./TracesTable.component-v8FAYs_j.js";import"./IconAlertCircle-BKZdSC5f.js";import"./IconFileDescription-BKfQSike.js";import"./IconTimeline-BvsfcN9Y.js";import"./IconInfoCircle-B0aEKl0P.js";function b(){const e=ae(),r=p(Te),c=p(ne),D=p(oe),B=p(ce),F=p(pe),L=p(me),I=ie(L,{pollingInterval:r>0?r:void 0}),j=I.data,{isLoading:E,isFetching:k,isError:m,error:A,refetch:U}=I,M=o.useCallback(t=>{e(le(t))},[e]),$=o.useCallback(t=>{e(de({column:t.columnAccessor,direction:t.direction}))},[e]),z=o.useCallback(t=>{e(ue(t))},[e]),Y=o.useCallback(t=>{e(he(t))},[e]),O=o.useCallback(()=>{e(ge())},[e]),Q=E&&!((j?.items?.length??0)>0);return o.useEffect(()=>{const t=m?je(A):null;if(m){const N=t?` (${t})`:"";e(_e({id:"resources-fetch",message:`Unable to refresh resources${N}. The table may be out of date until the connection recovers.`,tone:"error"}));return}!E&&!k&&e(P({id:"resources-fetch"}))},[e,A,m,k,E]),o.useEffect(()=>()=>{e(P({id:"resources-fetch"}))},[e]),s.jsxs(Ee,{gap:"md",children:[s.jsx(ke,{order:1,children:"Resources"}),s.jsx(Ae,{placeholder:"Search by Resources ID",value:c,onChange:t=>M(t.currentTarget.value),leftSection:s.jsx(Ie,{size:16}),"data-testid":"resources-search-input",w:"100%",style:{maxWidth:360}}),Q?s.jsx(K,{height:360,radius:"md"}):s.jsx(we,{resourcesList:j?.items,totalRecords:j?.total??0,isFetching:k,isError:m,error:A,searchTerm:c,sort:F,page:D,recordsPerPage:B,onSortStatusChange:$,onPageChange:z,onRecordsPerPageChange:Y,onResetFilters:O,onRefetch:U,renderRowExpansion:({resources:t})=>s.jsx(be,{resources:t})})]})}b.__docgenInfo={description:"",methods:[],displayName:"ResourcesPage"};const sr={title:"Pages/ResourcesPage",component:b,parameters:{layout:"fullscreen",chromatic:{modes:V}}},a=G,H=[{resourcesId:"rs-a1b2c3d4e5f6",version:3,createTime:a-168*3600,updateTime:a-3600,resources:{main_llm:{resource_type:"llm",endpoint:"https://api.openai.com/v1",model:"gpt-4",sampling_parameters:{temperature:.7,max_tokens:2048}},backup_llm:{resource_type:"llm",endpoint:"https://api.anthropic.com/v1",model:"claude-3-opus-20240229",sampling_parameters:{temperature:.5,max_tokens:4096}},greeting_template:{resource_type:"prompt_template",template:"Hello {name}! Welcome to {service}.",engine:"f-string"}}},{resourcesId:"rs-f6e5d4c3b2a1",version:2,createTime:a-336*3600,updateTime:a-48*3600,resources:{production_llm:{resource_type:"llm",endpoint:"https://api.openai.com/v1",model:"gpt-3.5-turbo",sampling_parameters:{temperature:.8,max_tokens:1024}},system_prompt:{resource_type:"prompt_template",template:"You are a helpful assistant. {context}",engine:"f-string"}}},{resourcesId:"rs-1234567890ab",version:1,createTime:a-720*3600,updateTime:a-360*3600,resources:{legacy_llm:{resource_type:"llm",endpoint:"https://api.openai.com/v1",model:"gpt-3.5-turbo-0301",sampling_parameters:{temperature:.9}}}},{resourcesId:"rs-abcdef123456",version:5,createTime:a-1440*3600,updateTime:a-12*3600,resources:{eval_llm:{resource_type:"llm",endpoint:"https://api.anthropic.com/v1",model:"claude-3-sonnet-20240229",sampling_parameters:{temperature:.3,max_tokens:2048}},judge_template:{resource_type:"prompt_template",template:`Evaluate the following response: {response}

Criteria: {criteria}`,engine:"jinja2"},dataset:{resource_type:"dataset",path:"s3://my-bucket/eval-data/v1/",format:"jsonl"}}},{resourcesId:"rs-999888777666",version:1,createTime:a-48*3600,updateTime:a-48*3600,resources:{test_llm:{resource_type:"llm",endpoint:"http://localhost:8080/v1",model:"local-model",sampling_parameters:{temperature:1}}}}],T=R(H);function q(e){return fe({config:{...ye,baseUrl:J,autoRefreshMs:0,...e},rollouts:xe,resources:ve})}function n(e){const r=q(e);return s.jsx(C,{store:r,children:s.jsxs(s.Fragment,{children:[s.jsx(b,{}),s.jsx(Re,{}),s.jsx(W,{})]})})}function Pe(e){const r=q(e),c=Se([{path:"/",element:s.jsx(se,{config:{baseUrl:r.getState().config.baseUrl,autoRefreshMs:r.getState().config.autoRefreshMs}}),children:[{path:"/resources",element:s.jsx(b,{})}]}],{initialEntries:["/resources"]});return s.jsx(C,{store:r,children:s.jsxs(s.Fragment,{children:[s.jsx(te,{router:c}),s.jsx(W,{})]})})}const i={render:()=>n(),parameters:{msw:{handlers:T}}},l={name:"Within AppLayout",render:()=>Pe(),parameters:{msw:{handlers:T}}},d={render:()=>n(),parameters:{msw:{handlers:T}},play:async({canvasElement:e})=>{const r=X(e);await r.findByText("rs-a1b2c3d4e5f6");const c=r.getByPlaceholderText("Search by Resources ID");await Z.type(c,"rs-abcdef123456"),await ee(()=>{if(r.queryByText("rs-a1b2c3d4e5f6"))throw new Error("Expected search to filter out non-matching resources");if(!r.queryByText("rs-abcdef123456"))throw new Error("Expected matching resource to remain visible")})}},u={render:()=>n(),parameters:{msw:{handlers:[S.get("*/v1/agl/resources",()=>w.json({items:[],limit:0,offset:0,total:0}))]}}},h={render:()=>n(),parameters:{msw:{handlers:[S.get("*/v1/agl/resources",()=>w.json({detail:"Internal server error"},{status:500}))]}}},g={render:()=>n(),parameters:{msw:{handlers:[S.get("*/v1/agl/resources",async()=>(await re(1200),w.json({detail:"Request timed out"},{status:504,statusText:"Timeout"})))]}}},_={render:()=>n(),parameters:{msw:{handlers:R([H[0]])}}},f={render:()=>n(),parameters:{msw:{handlers:[S.get("*/v1/agl/resources",()=>w.text("{ malformed json",{status:200,headers:{"Content-Type":"application/json"}}))]}}},y={render:()=>n(),parameters:{msw:{handlers:R(Array.from({length:50},(e,r)=>({resourcesId:`rs-generated-${r+1}`.padEnd(17,"0"),version:r%5+1,createTime:a-(50-r)*24*3600,updateTime:a-(50-r)*3600,resources:{llm:{resource_type:"llm",endpoint:`https://api.example.com/v${r%3+1}`,model:r%2===0?"gpt-4":"claude-3-opus",sampling_parameters:{temperature:.5+r%5*.1}}}})))}}},v={render:()=>n(),parameters:{msw:{handlers:R([{resourcesId:"rs-complex-123456",version:10,createTime:a-2160*3600,updateTime:a-600,resources:{primary_llm:{resource_type:"llm",endpoint:"https://api.openai.com/v1",model:"gpt-4-turbo-preview",sampling_parameters:{temperature:.7,top_p:.95,frequency_penalty:.1,presence_penalty:.1,max_tokens:4096}},fallback_llm:{resource_type:"llm",endpoint:"https://api.anthropic.com/v1",model:"claude-3-opus-20240229",sampling_parameters:{temperature:.7,max_tokens:4096,top_k:40}},embedding_model:{resource_type:"embedding",endpoint:"https://api.openai.com/v1",model:"text-embedding-3-large",dimensions:1536},system_prompt:{resource_type:"prompt_template",template:`You are an AI assistant. Context: {context}
User: {user_input}
Assistant:`,engine:"jinja2",variables:["context","user_input"]},training_dataset:{resource_type:"dataset",path:"s3://ml-datasets/training/v2/data.jsonl",format:"jsonl",size_bytes:1024e6,num_examples:5e4},validation_dataset:{resource_type:"dataset",path:"s3://ml-datasets/validation/v2/data.jsonl",format:"jsonl",size_bytes:1024e5,num_examples:5e3}}}])}}},x={render:()=>n({theme:"dark"}),parameters:{theme:"dark",msw:{handlers:T}}};i.parameters={...i.parameters,docs:{...i.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...i.parameters?.docs?.source}}};l.parameters={...l.parameters,docs:{...l.parameters?.docs,source:{originalSource:`{
  name: 'Within AppLayout',
  render: () => renderWithAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...l.parameters?.docs?.source}}};d.parameters={...d.parameters,docs:{...d.parameters?.docs,source:{originalSource:`{
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
    await canvas.findByText('rs-a1b2c3d4e5f6');
    const searchInput = canvas.getByPlaceholderText('Search by Resources ID');
    await userEvent.type(searchInput, 'rs-abcdef123456');
    await waitFor(() => {
      if (canvas.queryByText('rs-a1b2c3d4e5f6')) {
        throw new Error('Expected search to filter out non-matching resources');
      }
      if (!canvas.queryByText('rs-abcdef123456')) {
        throw new Error('Expected matching resource to remain visible');
      }
    });
  }
}`,...d.parameters?.docs?.source}}};u.parameters={...u.parameters,docs:{...u.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/resources', () => HttpResponse.json({
        items: [],
        limit: 0,
        offset: 0,
        total: 0
      }))]
    }
  }
}`,...u.parameters?.docs?.source}}};h.parameters={...h.parameters,docs:{...h.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/resources', () => HttpResponse.json({
        detail: 'Internal server error'
      }, {
        status: 500
      }))]
    }
  }
}`,...h.parameters?.docs?.source}}};g.parameters={...g.parameters,docs:{...g.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/resources', async () => {
        await delay(1200);
        return HttpResponse.json({
          detail: 'Request timed out'
        }, {
          status: 504,
          statusText: 'Timeout'
        });
      })]
    }
  }
}`,...g.parameters?.docs?.source}}};_.parameters={..._.parameters,docs:{..._.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers([sampleResources[0]])
    }
  }
}`,..._.parameters?.docs?.source}}};f.parameters={...f.parameters,docs:{...f.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/resources', () => HttpResponse.text('{ malformed json', {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        }
      }))]
    }
  }
}`,...f.parameters?.docs?.source}}};y.parameters={...y.parameters,docs:{...y.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers(Array.from({
        length: 50
      }, (_, index) => ({
        resourcesId: \`rs-generated-\${index + 1}\`.padEnd(17, '0'),
        version: index % 5 + 1,
        createTime: now - (50 - index) * 24 * 3600,
        updateTime: now - (50 - index) * 3600,
        resources: {
          llm: {
            resource_type: 'llm',
            endpoint: \`https://api.example.com/v\${index % 3 + 1}\`,
            model: index % 2 === 0 ? 'gpt-4' : 'claude-3-opus',
            sampling_parameters: {
              temperature: 0.5 + index % 5 * 0.1
            }
          }
        }
      })))
    }
  }
}`,...y.parameters?.docs?.source}}};v.parameters={...v.parameters,docs:{...v.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers([{
        resourcesId: 'rs-complex-123456',
        version: 10,
        createTime: now - 90 * 24 * 3600,
        updateTime: now - 600,
        resources: {
          primary_llm: {
            resource_type: 'llm',
            endpoint: 'https://api.openai.com/v1',
            model: 'gpt-4-turbo-preview',
            sampling_parameters: {
              temperature: 0.7,
              top_p: 0.95,
              frequency_penalty: 0.1,
              presence_penalty: 0.1,
              max_tokens: 4096
            }
          },
          fallback_llm: {
            resource_type: 'llm',
            endpoint: 'https://api.anthropic.com/v1',
            model: 'claude-3-opus-20240229',
            sampling_parameters: {
              temperature: 0.7,
              max_tokens: 4096,
              top_k: 40
            }
          },
          embedding_model: {
            resource_type: 'embedding',
            endpoint: 'https://api.openai.com/v1',
            model: 'text-embedding-3-large',
            dimensions: 1536
          },
          system_prompt: {
            resource_type: 'prompt_template',
            template: 'You are an AI assistant. Context: {context}\\nUser: {user_input}\\nAssistant:',
            engine: 'jinja2',
            variables: ['context', 'user_input']
          },
          training_dataset: {
            resource_type: 'dataset',
            path: 's3://ml-datasets/training/v2/data.jsonl',
            format: 'jsonl',
            size_bytes: 1024000000,
            num_examples: 50000
          },
          validation_dataset: {
            resource_type: 'dataset',
            path: 's3://ml-datasets/validation/v2/data.jsonl',
            format: 'jsonl',
            size_bytes: 102400000,
            num_examples: 5000
          }
        }
      }])
    }
  }
}`,...v.parameters?.docs?.source}}};x.parameters={...x.parameters,docs:{...x.parameters?.docs,source:{originalSource:`{
  render: () => renderWithStore({
    theme: 'dark'
  }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: defaultHandlers
    }
  }
}`,...x.parameters?.docs?.source}}};export{v as ComplexResources,x as DarkTheme,i as Default,u as EmptyState,y as ManyResources,f as ParseFailure,g as RequestTimeout,d as Search,h as ServerError,_ as SingleResource,l as WithSidebarLayout,sr as default};
