const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const context={window:{},document:{currentScript:null}};
vm.runInNewContext(fs.readFileSync('app/static/system-data-desk.js','utf8'),context);
const normalise=context.window.SystemDataReadiness.normalise;
test('GullMate governance projects existing metadata without inventing values or changing source hashes',()=>{
 const s={simulation:{sequence:2},metadata:{payload_sha256:'original-hash'},quality:{completeness_rate:1},governance:{production_authority:false},energy:{grid_demand_kw:0},port:{},yard:{},gate:{},fleet_summary:{}};
 const c={domains:[{id:'energy_carbon',required_fields:['grid_demand_kw'],source_system:'EMS'},{id:'governance',required_fields:['sequence','payload_sha256','completeness_rate','production_authority'],source_system:'audit'}],calibration_sources:[]};
 const before=JSON.stringify(s),d=normalise('xiaoyi',s,c);
 assert.equal(d.fieldCount,5);assert.equal(d.structureCount,5);assert.equal(d.valuedCount,5);
 assert.equal(d.fields.find(f=>f.id==='governance.payload_sha256').value,'original-hash');
 assert.equal(JSON.stringify(s),before);assert.equal(d.production_authority,false);
});
test('energy expected contract count survives missing signals; zero and false remain values',()=>{
 const s={schema_version:'v1',signals:{a:{value:0,unit:'kW'},b:{value:false,unit:'boolean'}},dataset:{},quality:{}};
 const d=normalise('energy',s,{field_count:3,required_field_attributes:['value','unit'],source_adapter_mapping:{}});
 assert.equal(d.fieldCount,3);assert.equal(d.valuedCount,2);assert.equal(d.structureCount,2);
 assert.equal(d.readouts[0][1],undefined);
});
test('malacca uses its own source values and does not promote simulated access to authority',()=>{
 const d=normalise('malacca',{schema_version:'v1',operationalTelemetry:{terminal:{queue:{value:0,unit:'vessel'},temperature:{value:null,unit:'C'}}},kpis:{queue_vessels:0},authority:{production_authority:true},calibration:{datasets:[]}}, {required_metadata:['value','unit'],adapter_slots:[]});
 assert.equal(d.fieldCount,2);assert.equal(d.valuedCount,1);assert.equal(d.production_authority,false);assert.equal(d.dispatch_allowed,false);
});
test('malformed or absent telemetry is an error instead of a successful empty report',()=>{
 for(const kind of ['xiaoyi','energy','malacca'])assert.throws(()=>normalise(kind,{},{}));
});
test('malacca checks every metadata attribute required by the fetched contract',()=>{
 const s={schema_version:'v1',operationalTelemetry:{energy:{load:{value:0,unit:'kW',event_time:'time',source_type:'physics_simulation',source_id:'engine'}}},calibration:{datasets:[]}};
 const d=normalise('malacca',s,{required_metadata:['value','unit','event_time','source_type','source_id','trace_id']});
 assert.equal(d.structureCount,0);assert.equal(d.valuedCount,1);
 assert.equal(d.groups[0].missing[0],'energy.load');
});
