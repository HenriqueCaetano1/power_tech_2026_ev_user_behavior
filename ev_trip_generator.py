class EvUser:
    def __init__(self,home,work,index,dm_user,v2g=0,parking_behavior=0,
                 charging_behavior=0,private_charge=0,vehicle_type=1,soc=0.8,initial_time=8):
        #ev_user=EvUser(home,work,user,dm[user],c_v2g[user],c_parking_behavior[user],
                       #c_charging_behavior[user],c_private_charge[user],c_vehicle_type[user])
        ### USER BEHAVIOR ASPECTS
        self.v2g=v2g #0: not v2g, 1: centralized v2g, 2: descentralized v2g
        self.private_charge=private_charge # 0: no private charging, 1: home, 2: work, 3: both work and home
        self.charging_behavior=charging_behavior # 0 wait in private charging location; 1 is go to nearest evcs
        self.parking_behavior=parking_behavior #0 is'charge' based (parking_time=charging_time); 1 is 'loc' based (parking time depends on the destination type: 'H, 'W' or 'P')
        self.private_trip=-1 #-1 if it does private before work; 1 if it does after work; 0 if no preference
        self.soc_min=0.2 #minimum soc (if the soc gets lower than that in a trip, user will recharge)
        self.soc_max=0.8 #maximum soc (if the user is trip based, he will leave the evcs as soon as this level is achieved)
        
        self.vehicle_type=vehicle_type #0 for fuel based, 1 for ev
        self.dm_user=dm_user
        self.home=home
        self.work=work
        
        self.trip_chain_df=pd.DataFrame(columns=['origin','destination','initial_time','final_time','trip_soc','destination_type'])
        self.parking_chain_df=pd.DataFrame(columns=['ev_idx','local','arrival_time','departure_time','arrival_soc','departure_soc'])
        self.ev_consumption=0
        self.fuel_consumption=0
        self.initial_soc=soc
        self.initial_time=initial_time
        self.soc=np.random.uniform(low=0.2,high=0.8) #initial SoC
        self.time=initial_time
        self.index=index
        
    

    def generate_trip_locations(self,weekday_user,n_trips_user):

        trip_chain='H' #starting at home
        # generate trip chain for a single user and a single day
        if(weekday_user==1): #weekday
            available_trips=n_trips_user-2
            trips_before_work=np.random.randint(0,available_trips+1)
            trips_after_work=available_trips-trips_before_work
            for i in range(trips_before_work):
                trip_chain+='P'
            trip_chain+='W'
            for i in range(trips_after_work):
                trip_chain+='P'
            trip_chain+='H'
        else: #weekend, the user does not go to work, instead do only private trips
            for i in range(n_trips_user-2):
                trip_chain+='P'
            trip_chain+='H'
        
        return trip_chain
    
    def simulate_parking(self,G,loc_parking_time,charging_node):
        #print('entrou parking')
        initial_time=self.time
        initial_soc=self.soc
        vehicle_type=self.vehicle_type

        if(vehicle_type==0): #fuel based, the user will fill the tank
            final_soc=1
            final_time=initial_time+5/60 #visit will only take 5 minutes
            parking_time=5
        else: # ev
            if(self.parking_behavior==1): # loc based -> parking time depends on the destination type (H, W or P)
                parking_time=loc_parking_time
            else: #charge based -> calculate time to get to maximum SoC
                parking_time=((self.soc_max-initial_soc)*max_cap_battery)/(consumption_rate*60) #parking time in hours
                #print('test parking time ',parking_time)
            final_soc=min(1,initial_soc+(consumption_rate*parking_time)/max_cap_battery*60) #convert hours to minutes and verify the charging
            final_time=initial_time+parking_time

        
        #print('soc charging',final_soc)
        self.soc=final_soc
        self.time=final_time

        #pd.DataFrame(columns=['local','arrival_time','departure_time'])
        self.parking_chain_df.loc[len(self.parking_chain_df)]=[self.index,charging_node,initial_time,final_time,initial_soc,final_soc]
        return parking_time

    def simulate_trip(self,G,origin,destination,destination_type):
        if(self.private_charge in [1,3] and destination_type=='H'): #user is leaving home -> SoC=1
            initial_soc=1
        else:
            initial_soc=self.soc
        initial_time=self.time
        #print(self.time)
        vehicle_type=self.vehicle_type
        path_time=nx.shortest_path_length(G,origin,destination,weight='travel_time')/(60*60)  # to be in hours
        path_length=nx.shortest_path_length(G,origin,destination,weight='length')
        #print(path_length)
        if(vehicle_type==0): #fuel based
            trip_soc=1e-3*path_length/(fuel_consumption_rate*max_fuel_capacity) 
            final_soc=max(initial_soc-trip_soc,0)
            fuel_consumption=(initial_soc-final_soc)*max_fuel_capacity*path_time #since path_time is in hours and battery capacity is in kw, final value will be in kwh
            ev_consumption=0
        else: #ev
            trip_soc=1e-3*path_length*consumption_per_kilometer/max_cap_battery
            final_soc=max(initial_soc-trip_soc,0)
            fuel_consumption=0
            ev_consumption=max(min(initial_soc-final_soc,1),0)*max_cap_battery*path_time #since path_time is in hours and battery capacity is in kw, final value will be in kwh

        final_time=initial_time+path_time
        self.trip_chain_df.loc[len(self.trip_chain_df)]=[origin,destination,initial_time,final_time,trip_soc,destination_type]
        self.ev_consumption+=ev_consumption
        self.fuel_consumption+=fuel_consumption
        self.soc=final_soc
        self.time=final_time

        return [final_soc,final_time,ev_consumption,fuel_consumption]
    
class EvStation:
    def __init__(self,node_location,num_spots=10,time_horizon=24,time_scale=0.25):
        self.node_location=node_location
        self.num_spots=num_spots

        num_time_slices=int(time_horizon/time_scale)
        self.num_time_slices=num_time_slices
        self.time_horizon=time_horizon
        self.time_scale=time_scale


        self.spots=np.zeros((num_time_slices,num_spots)) # spots(i,j) means that ev i is allocated there at time j
        self.available=[1]*num_time_slices # all spots starts available

        for time_idx in range(num_time_slices):
            for spot_idx in range(num_spots):
                self.spots[time_idx][spot_idx]=-1

    def allocate_ev(self,vehicle,initial_time,final_time): #allocate vehicle at given EvStation at time t
        #print('oii')
        #[i for i in range(len(new_city.evcs_list[0].spots[0])) if new_city.evcs_list[0].spots[0][i]==0]
        initial_time_slice=int(np.round(initial_time/self.time_scale))
        final_time_slice=int(np.round(final_time/self.time_scale))

        available_idx=[i for i in range(len(self.spots[initial_time_slice])) if self.spots[initial_time_slice][i]==-1]

        to_allocate_idx=np.random.choice(available_idx)
        
        for t in range(initial_time_slice,final_time_slice):
            self.spots[t][to_allocate_idx]=vehicle.index

        if(len(available_idx)==1): #if there was only one available spot, this evcs is now not available between initial time and final time
            for t in range(initial_time_slice,final_time_slice):
                self.available[t]=0
    

        
class City:
    def __init__(self,city_name='Esslingen',n_ev=None,n_days=7,p_ev=0.5,num_evcs=None,num_fuel_stations=None,
                 p_v2g=None,p_parking_behavior=None,p_charging_behavior=None,p_private_charging=None,ev_list=None,evcs_list=None):
        self.city_info_dict=retrieve_city_info(city_name)
        
        if(n_ev==None):
            self.n_ev=int(0.1*len(self.city_info_dict['residential_nodes']))
            print('n_ev:', self.n_ev)
        else:
            self.n_ev=n_ev # number of EVs
        self.n_days=n_days
        self.p_ev=p_ev
        self.time_horizon=24
        self.time_slice=0.25

        #custom probabilities for user behavior
        if(p_v2g==None):
            p_v2g=[1/3,1/3,1/3]

        if(p_parking_behavior==None):
            p_parking_behavior=[1/2,1/2]

        if(p_charging_behavior==None):
            p_charging_behavior=[1/2,1/2]

        if(p_private_charging==None):
            p_private_charging=[1/4,1/4,1/4,1/4]

        #initiate all users in the city
        # first the parameters that are set only once per user
        dm=1000*a_dm*np.random.weibull(a=b_dm,size=self.n_ev) # daily mileage (for the whole day), convertido em metros


        c_v2g=np.random.choice(np.arange(3),p=p_v2g,size=self.n_ev) # choice of doing v2g. 0=no vg, 1=centralized v2g, 2=descentralized v2g
        #c_v2g=np.random.randint(low=0,high=3,size=n_ev) 

        c_parking_behavior=np.random.choice(np.arange(2),p=p_parking_behavior,size=self.n_ev)

        c_charging_behavior=np.random.choice(np.arange(2),p=p_charging_behavior,size=self.n_ev)

        c_private_charge=np.random.choice(np.arange(4),p=p_private_charging,size=self.n_ev) # choice of private charging. 0=no, 1=work, 2=home, 3=both
        
        c_vehicle_type=np.random.binomial(n=1,p=p_ev,size=self.n_ev) # 0 is fuel based, 1 is ev
        
        c_weekday=np.random.binomial(n=1,p=5/7,size=n_days) # verify if it is a weekday or not

        self.G=self.city_info_dict['graph']
        self.commercial_nodes=self.city_info_dict['commercial_nodes']
        self.residential_nodes=self.city_info_dict['residential_nodes']
    
        if(num_evcs==None):
            self.evcs_nodes=self.city_info_dict['evcs_nodes']
        else:
            self.evcs_nodes=np.random.choice(self.city_info_dict['evcs_nodes'],num_evcs)

        if(num_fuel_stations==None):
            self.fuel_nodes=self.city_info_dict['fuel_nodes']
        else:
            self.fuel_nodes=np.random.choice(self.city_info_dict['fuel_nodes'],num_fuel_stations)



        if(ev_list==None):
            self.ev_list=[]
            for user in range(self.n_ev):
                to_break=False
                while(to_break==False):
                    home=np.random.choice(self.residential_nodes)
                    work=np.random.choice(self.commercial_nodes)
                    if((nx.has_path(self.G,home,work)) and (nx.shortest_path_length(self.G,home,work,weight='length')<dm[user]/2)): # you must be able to at least do H->W->H with your daily mileage
                        to_break=True
                ev_user=EvUser(home,work,user,dm[user],c_v2g[user],c_parking_behavior[user],
                            c_charging_behavior[user],c_private_charge[user],c_vehicle_type[user])
                self.ev_list.append(ev_user)
        else:
            self.ev_list=copy.deepcopy(ev_list)
            self.n_ev=len(ev_list)


        if(evcs_list==None):
            self.evcs_list=[]
            for evcs_node in self.evcs_nodes:
                evcs=EvStation(evcs_node)
                self.evcs_list.append(evcs)
        else:
            self.evcs_list=copy.deepcopy(evcs_list)
            self.evcs_nodes=[charging_station.node_location for charging_station in evcs_list]


    def find_available_fuel(self,G,origin):
        destination_list=[fuel for fuel in self.fuel_nodes]
        shortest_path_length=1e9
        for destination in destination_list:
            path_length=nx.shortest_path_length(G,origin,destination,weight='travel_time')
            if(path_length<shortest_path_length):
                shortest_node=destination
                shortest_path_length=path_length

        charging_length=nx.shortest_path_length(G,origin,shortest_node,weight='length')

        return [shortest_node,shortest_path_length,charging_length]

    def find_available_evcs(self,G,origin,t):
        time_idx=int((t/self.time_slice))
        destination_list=[evcs.node_location for evcs in self.evcs_list if evcs.available[time_idx]==1]
        #print(destination_list)
        shortest_path_length=1e9
        for destination in destination_list:
            try:
                path_length=nx.shortest_path_length(G,origin,destination,weight='travel_time')
                #print(path_length)
                if(path_length<shortest_path_length):
                    shortest_node=destination
                    shortest_path_length=path_length
            except:
                pass

        charging_length=nx.shortest_path_length(G,origin,shortest_node,weight='length')

        return [shortest_node,shortest_path_length,charging_length]




    def find_evcs_and_allocate_ev(self,vehicle,charging_node,initial_time,final_time):
        #print('eae man',charging_node)
        if(charging_node in self.evcs_nodes): # if node is a evcs, allocate to the proper parking slot. otherwise it is private charging 
            for evcs in self.evcs_list:
                if(evcs.node_location==charging_node):
                    chosen_evcs=evcs
                    break
            
            chosen_evcs.allocate_ev(vehicle,initial_time,final_time)


    def generate_trip_chains(self,weekday=1,to_reset=False):
        # generate a trip for each user in the city, for a single day
        # only the spent SoC for each trip will be considered. not the charged SoC (this will be optimized only on the charging schedule)
        city_info_dict=self.city_info_dict
        G=city_info_dict['graph']
        commercial_nodes=city_info_dict['commercial_nodes']
        residential_nodes=city_info_dict['residential_nodes']
        fuel_nodes=city_info_dict['fuel_nodes']
        evcs_nodes=city_info_dict['evcs_nodes']

        if(to_reset==True):
            for evcs_idx in range(len(self.evcs_list)):
                evcs=self.evcs_list[evcs_idx]
                for time_idx in range(evcs.num_time_slices):
                    for spot_idx in range(evcs.num_spots):
                        evcs.spots[time_idx][spot_idx]=-1


        for vehicle_idx in range(len(self.ev_list)):
            vehicle=self.ev_list[vehicle_idx]
            if(to_reset==True):
                vehicle.parking_chain_df=pd.DataFrame(columns=['ev_idx','local','arrival_time','departure_time','arrival_soc','departure_soc'])
            vehicle.time=copy.deepcopy(vehicle.initial_time)
            vehicle.time=np.random.normal(loc=mu_stt,scale=sigma_stt) #starting travel time
            pt_work_user=np.random.gumbel(mu_pt_w,sigma_pt_w) #Parking time for work location
            pt_other_user=np.random.gumbel(mu_pt_o,sigma_pt_o) #Parking time for other locations
            n_trips_user=np.random.randint(low=2,high=5) #number of trips per day between 2 and 4 accorging to nhts2017
            trip_locations='H' #starting at home
            # generate trip chain for a single user and a single day
            if(weekday==1): #dia util
                available_trips=n_trips_user-2
                if(vehicle.private_trip==-1): #private trip mostly before work
                    trips_before_work=available_trips
                    trips_after_work=0
                elif(vehicle.private_trip==1): #private trip mostly after work
                    trips_before_work=0
                    trips_after_work=available_trips
                else: #no preference
                    trips_before_work=np.random.randint(0,available_trips+1)
                    trips_after_work=available_trips-trips_before_work
                for i in range(trips_before_work):
                    trip_locations+='P'
                trip_locations+='W'
                for i in range(trips_after_work):
                    trip_locations+='P'
                trip_locations+='H'
            else:
                for i in range(n_trips_user-2):
                    trip_locations+='P'
                trip_locations+='H'
            
            individual_trip_list=[(trip_locations[i],trip_locations[i+1]) for i in range(len( trip_locations)-1)]

            for (origin_type,destination_type) in individual_trip_list:
                # first we define origin and departure of the trip
                to_break=False
                while(to_break==False):
                    if(origin_type=='H'):
                        origin=vehicle.home
                    elif(origin_type=='W'):
                        origin=vehicle.work
                    else:
                        origin=np.random.choice(G.nodes)

                    if(destination_type=='H'):
                        destination=vehicle.home
                        parking_time=pt_work_user
                    elif(destination_type=='W'):
                        destination=vehicle.work
                        parking_time=pt_other_user
                    else:
                        destination=np.random.choice(G.nodes)
                    if(nx.has_path(G,origin,destination)):
                        to_break=True
                
                
                path_time=nx.shortest_path_length(G,origin,destination,weight='travel_time')/(60*60)  # to be in hours
                path_length=nx.shortest_path_length(G,origin,destination,weight='length')

                #to-do: check feasibility of the trip. otherwise user will have to charge. and this may generate more trips (charging at origin)
                #to-do: charging at origin (private or public depending on the user choice)
                if(vehicle.vehicle_type==0): # fuel
                    kilometers_left=((vehicle.soc-vehicle.soc_min)*max_fuel_capacity)*fuel_consumption_rate #how many kilometers can you travel with current fuel (considering range anxiety)
                else: # ev
                    kilometers_left=((vehicle.soc-vehicle.soc_min)*max_cap_battery)/consumption_per_kilometer #how many kilometers can you travel with current battery (considering range anxiety)
                
                #print('kilometers left',kilometers_left,1e-3*path_length)
                #print('kilometers comparison',kilometers_left,path_length*1e-3)
                if(kilometers_left<=path_length*1e-3): #must either find charging location or wait for charging at current location
                    if(vehicle.vehicle_type==1):
                        required_soc=((1e-3*path_length)*consumption_per_kilometer)/max_cap_battery
                        parking_time=(required_soc*max_cap_battery)/consumption_rate #required parking time in minutes
                    #p_v2g=[1-p_value,0,p_value] #0: not v2g, 1: centralized v2g, 2: descentralized v2g
                    #p_parking_behavior=[1/2,1/2]#0 is'charge' based (parking_time=charging_time); 1 is 'loc' based (parking time depends on the destination type: 'H, 'W' or 'P')
                    #p_charging_behavior=[1/2,1/2]# 0 wait in private charging location; 1 is go to nearest evcs
                    #p_private_charging=[1,0,0,0] # 0: no private charging, 1: home, 2: work, 3: both work and home
                    if(vehicle.charging_behavior==0 and origin_type in ['H','W'] and vehicle.private_charge in [1,3]):
                        initial_time=copy.deepcopy(vehicle.time)
                        real_parking_time=vehicle.simulate_parking(G,parking_time,charging_node=origin) #origin will be 'H' or 'W'
                        self.find_evcs_and_allocate_ev(vehicle=vehicle,charging_node=origin,initial_time=initial_time,final_time=initial_time+real_parking_time)
                        
                    else: #otherwise find a new place to travel and charge
                        if(vehicle.vehicle_type==0): #fuel
                            [charging_node,travelling_time,travelling_length]=self.find_available_fuel(G,origin)
                        else: #ev
                            [charging_node,travelling_time,travelling_length]=self.find_available_evcs(G,origin,vehicle.time)
                        #arriving at destination
                        vehicle.simulate_trip(G,origin,charging_node,destination_type='cs')

                        origin=charging_node #new origin for next trip simulation
                        #charging at destination
                        initial_time=copy.deepcopy(vehicle.time)
                        real_parking_time=vehicle.simulate_parking(G,parking_time,charging_node=charging_node)
                        self.find_evcs_and_allocate_ev(vehicle=vehicle,charging_node=origin,initial_time=initial_time,final_time=initial_time+real_parking_time)

                
                #arriving at destination
                vehicle.simulate_trip(G,origin,destination,destination_type=destination_type)

                #charging at destination (private charging, home or work)
                if(vehicle.charging_behavior==0 and destination_type in ['H','W'] and vehicle.private_charge in [1,3]):
                    initial_time=copy.deepcopy(vehicle.time)
                    real_parking_time=vehicle.simulate_parking(G,parking_time,charging_node=destination)
                    self.find_evcs_and_allocate_ev(vehicle=vehicle,charging_node=origin,initial_time=initial_time,final_time=initial_time+real_parking_time)


                #print(single_fuel_consumption)
                
    def generate_full_parking_df(self):
        df_list=[self.ev_list[i].parking_chain_df for i in range(len(self.ev_list))]
        result=pd.concat(df_list)
        result.index=range(len(result))

        self.full_parking_df=result

        time_steps=range(int(np.round(self.time_horizon/self.time_slice)))

        #self.x_ev=np.zeros((len(self.full_parking_df.index),len(self.evcs_list),len(time_steps))) # x(i,j,t)=1 (0) means that the ev i is (is not) at evcs j at time t

        x_ev_list=[]
        


        #soc_a=np.zeros((len(new_city.ev_list),len(new_city.evcs_list),len(time_steps))) # soc_a(i,j,t)=k means that the ev i arrived in evcs j at time t with soc k
        self.soc_a=np.array(self.full_parking_df.arrival_soc)
        
        #soc_d=np.zeros((len(new_city.ev_list),len(new_city.evcs_list),len(time_steps))) # soc_d(i,j,t)=k means that the ev i left from evcs j at time t with soc k
        self.soc_d=np.array(self.full_parking_df.departure_soc)

        private_parking_list=[]

        for parking_idx in range(len(self.full_parking_df)):
            evcs_node=int(self.full_parking_df.at[parking_idx,'local'])
            evcs_node_location_list=[evcs.node_location for evcs in self.evcs_list]

            if(evcs_node in evcs_node_location_list):
                private_parking_list.append(False)
                x_ev_list.append([])
                evcs_parked_idx=evcs_node_location_list.index(evcs_node)
                t_a=self.full_parking_df.at[parking_idx,'arrival_time']
                t_d=self.full_parking_df.at[parking_idx,'departure_time']

                for evcs_idx in range(len(self.evcs_list)):
                    x_ev_list[-1].append([])
                    if(evcs_idx==evcs_parked_idx):
                        for time in time_steps:
                            if((time*self.time_slice > t_a) and (time*self.time_slice < t_d)):
                                x_ev_list[-1][-1].append(1)
                                #self.x_ev[parking_idx][evcs_idx][time]=1
                            else:
                                x_ev_list[-1][-1].append(0)
                    else:
                        for time in time_steps:
                            x_ev_list[-1][-1].append(0)
            else:
                private_parking_list.append(True)
        
    
        self.x_ev=np.array(x_ev_list)
        #print(private_parking_list)
        #print(len(self.full_parking_df))
        self.full_parking_df=self.full_parking_df.assign(private_parking=private_parking_list)

    def calculate_evcs_occupation(self,mean_value=True):
        occupation_list=[]
        for evcs in self.evcs_list:
            occupation_list.append([])
            spots=evcs.spots
            for time_idx in range(spots.shape[0]):
                occupation_list[-1].append(sum([spots[time_idx][spot_idx]!=-1 for spot_idx in range(spots.shape[1])])/spots.shape[1])

        if(mean_value==True):
            mean_occupation_list=[np.mean([occupation_list[evcs_idx][time_idx] for evcs_idx in range(len(self.evcs_list))]) for time_idx in range(self.evcs_list[0].spots.shape[0])]
            return mean_occupation_list
        else:
            return occupation_list


# step 1: defining the problem
class MyProblem(ElementwiseProblem):

    def __init__(self,city):
        self.num_time_slices=int(city.time_horizon/city.time_slice)
        self.x_ev=city.x_ev
        self.soc_a=city.soc_a
        self.soc_d=city.soc_d
        charging_rate_kw=6
        discharging_rate_kw=6
        self.time_slice=city.time_slice
        battery_capacity_kwh=60

        self.charging_rate=(charging_rate_kw*self.time_slice)/battery_capacity_kwh #charging rate in terms of SoC
        self.discharging_rate=(discharging_rate_kw*self.time_slice)/battery_capacity_kwh #discharging rate in terms of SoC
        self.soc_min_list=[]
        self.soc_max_list=[]
        self.alpha_sell=0.1
        self.beta_sell=0.3
        self.alpha_buy=self.alpha_sell/0.98 # The energy selling prices were considered as 98% of the energy buying prices 
        #(source: Charging scheduling in a workplace parking lot: Biobjective optimization approaches through predictive analytics of electric vehicle users' charging behavior
        self.beta_buy=self.alpha_buy

        xl=np.zeros(self.x_ev.shape[0]*self.x_ev.shape[2])
        xu=np.zeros(self.x_ev.shape[0]*self.x_ev.shape[2])

        array_idx=0
        for ev_idx in city.full_parking_df[city.full_parking_df.private_parking==False].ev_idx:
            vehicle=city.ev_list[int(ev_idx)]
            self.soc_min_list.append(vehicle.soc_min)
            self.soc_max_list.append(vehicle.soc_max)
            v2g_choice=vehicle.v2g
            for time_idx in range(self.x_ev.shape[2]):
                if(v2g_choice in [1,2]): #does v2g, either centralized or decentralized
                    xl[array_idx]=-1
                    xu[array_idx]=1
                else:
                    xl[array_idx]=0
                    xu[array_idx]=1
                array_idx+=1

        super().__init__(n_var=self.x_ev.shape[0]*self.x_ev.shape[2],
                         n_obj=3,
                         n_ieq_constr=2*self.x_ev.shape[0]*self.x_ev.shape[2], #soc min and max
                         xl=xl,
                         xu=xu, #to-do: consider the v2g restriction here
                         v_type=int)

    def _evaluate(self, x, out, *args, **kwargs):
        f1=0 # objective 1: user satisfaction
        f2=0 # objective 2: ev profit
        f3=0 #objective 3: minimum load variance (peak valley difference)

        g_list=[]

        soc_array=np.zeros((self.x_ev.shape[0],self.x_ev.shape[2]))

        for ev_idx in range(self.x_ev.shape[0]):
            soc=self.soc_a[ev_idx]
            for time_idx in range(self.num_time_slices):
                charging_decision=x[ev_idx*(self.num_time_slices)+time_idx]
                if(charging_decision==1):
                    soc+=self.charging_rate

                    # linear price function (source: Optimal Spatial–Temporal Scheduling for EVs With Stochastic Behaviors and Possible Inauthentic Information)
                    energy_to_buy=self.charging_rate*max_cap_battery*self.time_slice
                    f2-=self.alpha_buy*energy_to_buy+self.beta_buy #price in ($), negative because the user is buying 
                elif(charging_decision==-1):
                    soc-=self.discharging_rate
                    energy_to_sell=self.discharging_rate*max_cap_battery*self.time_slice
                    f2+=self.alpha_sell*energy_to_sell+self.beta_sell #price in ($), positive because the user is selling (profit)
                soc_array[ev_idx][time_idx]=soc
                g_list.append(self.soc_min_list[ev_idx]-soc)
                g_list.append(soc-self.soc_max_list[ev_idx])


            
            
            
            f1+=np.abs(soc-self.soc_d[ev_idx]) 


            f3_aux=0
            sum_soc=sum(sum(soc_array))
            for ev_idx in range(self.x_ev.shape[0]):
                for time_idx in range(self.num_time_slices):
                    f3_aux+=(soc_array[ev_idx][time_idx]-sum_soc/self.num_time_slices)**2 #to-do: verificar que a implementação está certa

            f3+=f3_aux/(self.x_ev.shape[0]*self.x_ev.shape[2])
        #print(f1+f2+f3)
        

        #f2 = (x[0]-1)**2 + x[1]**2




        #g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        #g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8
        out["F"] = [f1,-f2,f3] #dont forget that f2 is profit so it must be maximized
        #print(f1)
        #out["F"] = [f1] #dont forget that f2 is profit so it must be maximized
        out["G"] = g_list


def optimize_charging_scheduling(city,alg='nsga',n_gen=500):
    if(alg=='GA'):
        algorithm = GA(
        pop_size=20,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True)
    else:
        algorithm = NSGA2(pop_size=200,
                    n_offsprings=10,
                    sampling=IntegerRandomSampling(),
                    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )

    termination = get_termination("n_gen", n_gen)

    problem = MyProblem(city)

    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=False,
               return_least_infeasible=True)

    return res
