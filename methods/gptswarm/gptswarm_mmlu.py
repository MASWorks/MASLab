# The original repo also has “full” and “random” modes, but here we just use “optimized”.
import asyncio
import os
import time
import numpy as np
import pandas as pd
import shortuuid
import torch

from copy import deepcopy
from pathlib import Path
from termcolor import colored
from typing import Any,Iterator,Dict,List,Tuple

from .agents import IO,FinalDecision
from ..mas_base import MAS
from ..utils import load_config


class GPTswarm_MMLU(MAS):
    def __init__(self, general_config, method_config_name="config"):
        super().__init__(general_config)
        method_config_name = "config" if method_config_name is None else method_config_name
        self.method_config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml"))
        self.num_truthful_agents = self.method_config["num-truthful-agent"]
        self.num_iterations = self.method_config["num_iterations"]
        self.dataset_name = general_config['test_dataset_name']
        self.model_name_execute = general_config.get('optimize_execute_model_name','gpt-4o-mini-2024-07-18')
        self.results_path = f"results/{self.dataset_name}/gptswarm/{self.model_name_execute}"
        self.lr = self.method_config["lr"]
        self.used_agents = []
        self.agents = {"IO": IO}
        
        self.init_connection_probability = 0.5
        self.potential_connections = []

        self.decision_method = FinalDecision(env=self)
        
        self.memory: Dict[str, List[Dict[str, Any]]] = {}
        self.nodes = {}
        self.graphs = []
        self.input_nodes = []
        self.output_nodes = [self.decision_method]
        
        self.inference_flag = True

        self.add_node(self.decision_method)
        self.organize()

    def organize(self):
        agent_name_list = self.num_truthful_agents * ["IO"]
        for agent_name in agent_name_list:
            agent_instance = self.agents.get(agent_name)(env=self)    
            self.add_graph(agent_instance)
            self.used_agents.append(agent_instance)
        
        # Add bi-directional connections between all nodes of all agents (except for the decision nodes).
        for agent1 in self.used_agents:
            for agent2 in self.used_agents:
                if agent1 != agent2:
                    for node1 in agent1.nodes:
                        for node2 in agent2.nodes:
                            self.potential_connections.append((node1, node2)) # (from, to)

        for agent in self.used_agents:
            for node in agent.nodes:
                self.potential_connections.append((node, self.decision_method.id)) # (from, to)

        # Single scalar
        init_logit = torch.log(torch.tensor(self.init_connection_probability / (1 - self.init_connection_probability)))
        # The shape is one-dimensional and the length is len(self.potential_connections)
        init_tensor = torch.ones(len(self.potential_connections),requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)

        # A collection of node IDs for each agent
        node_ids = set([x for pair in self.potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = torch.nn.Parameter(order_tensor)

    def add_graph(self, graph):
        for node in graph.nodes.values():
            self.add_node(node)
        self.graphs.append(graph)
        self.input_nodes.extend(graph.input_nodes)
    
    def add_node(self, node):
        """
        Creates and adds a new node to the graph.
        If id is not provided, generates a unique id for the node.
        """
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=5)
        node.id = node_id
        self.nodes[node_id] = node
        return node 

    def check_cycle(self, new_node, target_nodes):
        # Once a loop is detected, True is returned.
        if new_node in target_nodes:
            return True
        for successor in new_node.successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def generate_graph(self,temperature: float = 1.0) -> Tuple[torch.Tensor]:
        # randomly generate graph
        log_probs = [torch.tensor(0.0, requires_grad=True)]
        _graph = deepcopy(self)
        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            out_node = _graph.nodes.get(potential_connection[0])
            in_node = _graph.nodes.get(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            if not _graph.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob
    
    def _generate_graph(self,edge_mask: torch.Tensor) -> Tuple[torch.Tensor]:
        _graph = deepcopy(self)
        for i, (potential_connection, is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
            out_node = _graph.nodes.get(potential_connection[0])
            in_node = _graph.nodes.get(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}):
                if is_edge:
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
        return _graph
    
    def inference(self, sample):
        query = sample.get("query")
        if not query:
            raise ValueError("Sample must contain a 'query' key.")
        self.inference_flag = True
        optimized_path = Path(self.results_path) / "best_workflow.npy"
        if optimized_path.exists():
            loaded_probs_npy = np.load(optimized_path)
            self.edge_logits = torch.from_numpy(loaded_probs_npy)
            edge_mask = self.edge_logits > 0.5
            graph = self._generate_graph(edge_mask)
            input_dict = {"task": query}
            response = graph._inference(input_dict)
            response = "\n".join(response)
        else:
            raise NotImplementedError("Best_workflow path does not exist!")
        return response
    

    def optimizing(self,val_dataset,batch_size: int = 4) -> torch.Tensor:
        # Here mmlu is optimized with dev.
        self.inference_flag = False
        optimized_path = Path(self.results_path) / "best_workflow.npy"
        if optimized_path.exists():
            print(colored("The optimal graph already exists!\n","red"))
            return
        print(colored("Optimizing swarm on MMLUDataset split dev...","light_yellow"))
        optimizer = torch.optim.Adam([self.edge_logits, self.order_params], lr=self.lr)
        
        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(val_dataset))
            while True:
                for idx in perm:
                    record = val_dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        edge_probs = None
        for i_iter in range(self.num_iterations):
            print(f"Iter {i_iter}", 80*'-')
            start_ts = time.time()
            raw_answers = []
            log_probs = []
            correct_answers = []

            for _, record in zip(range(batch_size), loader):

                graph, log_prob = self.generate_graph()

                demo_question = (f"{record['query']}\n")
                input_dict = {"task": demo_question}
                answer = graph._inference(input_dict)
                print(colored(answer,"light_cyan"))
                raw_answers.append(answer)
                log_probs.append(log_prob)
                
                correct_answer = record['gt'][1]
                
                assert isinstance(correct_answer, str), (
                    f"String expected but got {correct_answer} "
                    f"of type {type(correct_answer)} (2)" \
                    f" record={record}")
                correct_answers.append(correct_answer)

            

            print(f"Batch time {time.time() - start_ts:.3f}")

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            _num_correct = 0
            _num_total = 0

            for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
                if isinstance(raw_answer, list):
                    if len(raw_answer) > 0:
                        answer = raw_answer[0]
                    else:
                        answer = ""
                if not isinstance(answer, str):
                    raise Exception("Expected string")
                if len(answer) > 0:
                    answer = answer[0] # Try to format the answer by taking the first letter
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                
                is_correct = answer == correct_answer
                _num_correct += int(is_correct)
                _num_total += 1
                utility = _num_correct / _num_total
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss)

            print("utilities:", utilities)
            total_loss = torch.mean(torch.stack(loss_list))
            print("loss:", total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            print("Grad:", self.edge_logits.grad)
            optimizer.step()
            print("edge_logits:", self.edge_logits)
            edge_probs = torch.sigmoid(self.edge_logits)
            print("edge_probs:", edge_probs)
            print("end of iteration")

        print(colored("Done!","green"))
        edge_probs_np = self.edge_logits.detach().numpy()
        graph_path = self.results_path
        if not os.path.exists(graph_path):
            os.makedirs(graph_path) 
        dest = os.path.join(graph_path, "best_workflow.npy")
        np.save(dest, edge_probs_np)
        print(colored("Best graph saved!","light_yellow"))
    
    def _inference(self, inputs: Dict[str, Any], max_tries: int = 3, max_time: int = 600) -> List[Any]:
        
        def is_node_useful(node):
            if node in self.output_nodes:
                return True
            for successor in node.successors:
                if is_node_useful(successor):
                    return True
            return False
        
        useful_node_ids = [node_id for node_id, node in self.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(self.nodes[node_id].predecessors) for node_id in useful_node_ids}
        # Contains the IDs of all useful nodes with zero intake
        zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and node_id in useful_node_ids]

        for i, input_node in enumerate(self.input_nodes):
            node_input = deepcopy(inputs)
            input_node.inputs = [node_input]

        while zero_in_degree_queue:
            current_node_id = zero_in_degree_queue.pop(0)
            current_node = self.nodes[current_node_id]
            tries = 0
            while tries < max_tries:
                try:
                    asyncio.run(self.nodes[current_node_id].execute())
                    break
                except asyncio.TimeoutError:
                    print(f"Node {current_node_id} execution timed out, retrying {tries + 1} out of {max_tries}...")
                except Exception as e:
                    print(f"Error during execution of node {current_node_id}: {e}")
                    break
                tries += 1

            for successor in current_node.successors:
                if successor.id in useful_node_ids:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

        final_answers = []

        for output_node in self.output_nodes:
            output_messages = output_node.outputs
            # return all outputs
            if len(output_messages) > 0:
                final_answer = output_messages[-1].get("output", output_messages[-1])
                final_answers.append(final_answer)
            else:
                for output_message in output_messages:
                    final_answer = output_message.get("output", output_message)
                    final_answers.append(final_answer)

        if len(final_answers) == 0:
            final_answers.append("No answer since there are no inputs provided")
        return final_answers
    

    
         
    