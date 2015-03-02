
float4 getforce(float4 pos, float4 vel)
{
  float4 force;
  force.x = 0.0f;
  force.y = EPS_DOWN;
  force.z = 0.0f;
  force.w = 1.0f;
  return (force);
}

float goober(float prev)
{
  prev *= (MOD*MULT);
  return (fmod(prev,MOD)/MOD);
}

__kernel void VVerlet(__global float4* p, __global float4* v, __global float* r)
{
	unsigned int i = get_global_id(0);
	float4 force, zoom;

	for(int steps=0;steps < STEPS_PER_RENDER; steps++)
	{
    		force = getforce(p[i],v[i]);
   		v[i] += force*DELTA_T/2.0;
    		p[i] += v[i]*DELTA_T;
   		force = getforce(p[i],v[i]);
   		v[i] += force*DELTA_T/2.0;

		if(p[i].y < -0.5f) {
	        	zoom.x = r[i]+1.0f;
        		r[i] = goober(r[i]);
		        zoom.y = 0.4f*r[i] + 1.1f;
        		r[i] = goober(r[i]);
		        zoom.z = 1.0f*r[i]+1.0f;
		  	p[i] = zoom;
			v[i] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
		 	r[i] = goober(r[i]);
		}	
		if(0.5f*p[i].x + p[i].y-1.605f < 0.0f && (p[i].y > 0.6f)) {
			v[i].x = (FRICTION+0.5f*(1.0f+RESTITUTION1)*v[i].y);
			v[i].y = v[i].y*(RESTITUTION);
		}
		if(0.5f*p[i].x-p[i].y-0.64f > 0.0f && (p[i].y > 0.1f)) {		
			v[i].x = ((v[i].y*(1.0f+RESTITUTION)*0.5f) - FRICTION)*0.7f;
			v[i].y = v[i].y*RESTITUTION;
		}
	}
	p[i].w = 1.0f;
}
