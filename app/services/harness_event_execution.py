"""Discrete event execution of an explicitly simulated engineering resource plan.
It commits resources only inside this isolated workorder; never changes the site's shared plan.
"""
from datetime import datetime
import math

def _time(text):
 value=datetime.fromisoformat(text.replace('Z','+00:00'))
 if value.utcoffset() is None: raise ValueError('event clock requires timezone')
 return value

def execute_schedule(schedule):
 tasks=schedule['candidate_plan'];resources=schedule['resources'];epoch=_time(schedule['epoch'])
 capacities={r['resource_id']:float(r['capacity']) for r in resources}
 types={r['resource_id']:r['resource_type'] for r in resources}
 if any(not math.isfinite(x) or x<=0 for x in capacities.values()): raise ValueError('invalid resource capacity')
 ids=[t['task_id'] for t in tasks]
 if len(set(ids))!=len(ids) or len(ids)>100:raise ValueError('invalid task identifiers or plan size')
 state={i:'planned' for i in ids};occupancy={i:0. for i in capacities};ends={};events=[];queue=[]
 for task in tasks:
  a,b=_time(task['candidate_start']),_time(task['candidate_end'])
  if not epoch<=a<b or b>_time(task['deadline']): raise ValueError('invalid task interval or missed deadline')
  for resource in task['allocations']:
   q=float(resource['quantity'])
   if resource['resource_id'] not in capacities or not math.isfinite(q) or q<=0: raise ValueError('invalid resource commitment')
  queue += [(a,1,task),(b,0,task)]
 for at,start,task in sorted(queue,key=lambda r:(r[0],r[1],r[2]['task_id'])):
  name=task['task_id']
  if start:
   if state[name]!='planned':raise ValueError('task already started')
   for dep in task['predecessor_ids']:
    if state.get(dep)!='completed' or (at-ends[dep]).total_seconds()<900:raise ValueError('predecessor/handoff not completed')
   if task['chain_id']=='chain.01' and task['stage']=='port_arrival' and (at-epoch).total_seconds()<schedule['native_arrival_seconds']:
    raise ValueError('port work started before native vessel arrival')
   for row in task['allocations']:
    occupancy[row['resource_id']]+=row['quantity']
    if occupancy[row['resource_id']]>capacities[row['resource_id']]+1e-8:raise ValueError('capacity conflict during actual event execution')
   state[name]='running'
  else:
   if state[name]!='running':raise ValueError('task completed without start')
   for row in task['allocations']:occupancy[row['resource_id']]-=row['quantity']
   state[name]='completed';ends[name]=at
  events.append({'task_id':name,'chain_id':task['chain_id'],'stage':task['stage'],'event':'started' if start else 'completed',
    'engineering_time':at.isoformat(),'elapsed_seconds':(at-epoch).total_seconds(),'resource_occupancy':dict(occupancy)})
 if not all(v=='completed' for v in state.values()) or any(abs(v)>1e-8 for v in occupancy.values()):raise ValueError('unfinished tasks or leaked resources')
 return {'status':'completed_engineering_event_simulation','events':events,'task_states':state,'final_resource_occupancy':occupancy,
   'tasks_completed':len(tasks),'stage_coverage':sorted({t['stage'] for t in tasks}),'resources_used':sorted(set(types.values())),
   'production_authority':False,'site_resource_commitment':False,'evidence_class':'declared_engineering_fixture'}

def interval_controls(schedule,steps,step_hours,base):
 """Map time-weighted reservations to native V6 normalized control requests.
 Native deterministic projection still enforces its own safety constraints.
 """
 epoch=_time(schedule['epoch']);capacity={r['resource_id']:float(r['capacity']) for r in schedule['resources']}
 types={r['resource_id']:r['resource_type'] for r in schedule['resources']}
 mapping={'berth':3,'gate':7,'rail':8,'shore_power_bess':11,'maintenance_team':12,'pilot':13,'tug':14,'quay_crane':15,'horizontal_transport':16,'yard_block':17}
 result=[]
 for step in range(steps):
  begin,end=step*step_hours*3600,(step+1)*step_hours*3600;usage={r:0. for r in capacity}
  for task in schedule['candidate_plan']:
   a=(_time(task['candidate_start'])-epoch).total_seconds();b=(_time(task['candidate_end'])-epoch).total_seconds()
   fraction=max(0.,min(end,b)-max(begin,a))/(end-begin)
   for allocation in task['allocations']:usage[allocation['resource_id']]+=allocation['quantity']*fraction
  controls=list(base)
  for resource,amount in usage.items():
   typ=types[resource]
   if typ in mapping:controls[mapping[typ]]=max(-1.,min(1.,amount/capacity[resource]*2-1))
  result.append(controls)
 return result
