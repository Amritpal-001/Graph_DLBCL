import json
import os

for file in os.listdir("./graphs_int_cnt_proj"):
    if (".json" in file):
        with open("graphs_int_cnt_proj/" + file, 'r') as f:
            data = json.load(f)

        x_coords = [c[0] for c in data["coordinates"]]
        y_coords = [c[1] for c in data["coordinates"]]
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        x_med = x_coords[len(x_coords)//2]
        y_med = y_coords[len(y_coords)//2]
        g0 = 0
        g1 = 0
        g2 = 0
        g3 = 0
        g0_dict = {}
        g1_dict = {}
        g2_dict = {}
        g3_dict = {}
        g0_dict["x"] = []
        g1_dict["x"] = []
        g2_dict["x"] = []
        g3_dict["x"] = []
        g0_dict["y"] = data["y"]
        g1_dict["y"] = data["y"]
        g2_dict["y"] = data["y"]
        g3_dict["y"] = data["y"]
        g0_dict["coordinates"] = []
        g1_dict["coordinates"] = []
        g2_dict["coordinates"] = []
        g3_dict["coordinates"] = []
        g0_dict["edge_index"] = [[],[]]
        g1_dict["edge_index"] = [[],[]]
        g2_dict["edge_index"] = [[],[]]
        g3_dict["edge_index"] = [[],[]]
        quadrants = {}
        new_indices = {}
        for i in range(len(data["coordinates"])):
          c = data["coordinates"][i]
          if c[0] < x_med and c[1] >= y_med:
            quadrants[i] = 0
            new_indices[i] = g0
            g0_dict["x"].append(data["x"][i])
            g0_dict["coordinates"].append(data["coordinates"][i])
            g0 += 1
          elif c[0] >= x_med and c[1] >= y_med:
            quadrants[i] = 1
            new_indices[i] = g1
            g1_dict["x"].append(data["x"][i])
            g1_dict["coordinates"].append(data["coordinates"][i])
            g1 += 1
          elif c[0] >= x_med and c[1] < y_med:
            quadrants[i] = 2
            new_indices[i] = g2
            g2_dict["x"].append(data["x"][i])
            g2_dict["coordinates"].append(data["coordinates"][i])
            g2 += 1
          else:
            quadrants[i] = 3
            new_indices[i] = g3
            g3_dict["x"].append(data["x"][i])
            g3_dict["coordinates"].append(data["coordinates"][i])
            g3 += 1
        for i in range(len(data["edge_index"][0])):
          u = data["edge_index"][0][i]
          v = data["edge_index"][1][i]
          if quadrants[u] == 0 and quadrants[v] == 0:
            g0_dict["edge_index"][0].append(new_indices[u])
            g0_dict["edge_index"][1].append(new_indices[v])
          elif quadrants[u] == 1 and quadrants[v] == 1:
            g1_dict["edge_index"][0].append(new_indices[u])
            g1_dict["edge_index"][1].append(new_indices[v])
          elif quadrants[u] == 2 and quadrants[v] == 2:
            g2_dict["edge_index"][0].append(new_indices[u])
            g2_dict["edge_index"][1].append(new_indices[v])
          elif quadrants[u] == 3 and quadrants[v] == 3:
            g3_dict["edge_index"][0].append(new_indices[u])
            g3_dict["edge_index"][1].append(new_indices[v])
        with open("graphs_quads_int_cnt_proj/" + file[:-5] + "_0.json", "w") as outfile: 
            json.dump(g0_dict, outfile)
        with open("graphs_quads_int_cnt_proj/" + file[:-5] + "_1.json", "w") as outfile: 
            json.dump(g1_dict, outfile)
        with open("graphs_quads_int_cnt_proj/" + file[:-5] + "_2.json", "w") as outfile: 
            json.dump(g2_dict, outfile)
        with open("graphs_quads_int_cnt_proj/" + file[:-5] + "_3.json", "w") as outfile: 
            json.dump(g3_dict, outfile)

