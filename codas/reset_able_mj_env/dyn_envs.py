import os
import re
import time

import gym
from RLA.easy_log import logger


def get_new_gravity_env(variety: float, env_name):
    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)
    return env


def get_source_env(env_name):

    update_source_env(env_name)
    env = gym.make(env_name)
    return env


def get_new_density_env(variety: float, env_name):

    update_target_env_density(variety, env_name)
    env = gym.make(env_name)
    return env


def get_new_friction_env(variety: float, env_name):

    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)
    return env


def generate_xml_path():
    import sys
    path = sys.path

    xml_path = None

    for p in path:
        if "site-packages" in p[-14:] and "local" not in p:
            xml_path = p + '/gym/envs/mujoco/assets'

    assert xml_path is not None
    logger.info("gym xml file :{}".format(xml_path))
    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def update_xml(index, env_name):

    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'double' in env_name.lower():
        xml_name = "inverted_double_pendulum.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'reach' in env_name.lower():
        xml_name = "reacher.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "standup" in env_name.lower():
        xml_name = "humanoidstandup.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "striker" in env_name.lower():
        xml_name = "striker.xml"
    elif "swim" in env_name.lower():
        xml_name = "swimmer.xml"
    elif "throw" in env_name.lower():
        xml_name = "thrower.xml"
    elif "point" in env_name.lower():
        xml_name = "point.xml"
    elif "pendulum" in env_name.lower():
        xml_name = "inverted_pendulum.xml"
    elif "pusher" in env_name.lower():
        xml_name = "pusher.xml"
    elif "humanoid" in env_name.lower():
        xml_name = "humanoid.xml"
    else:
        raise RuntimeError("No available env named \'%s\'"%env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system('cp ./xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_gear(variety_degree, env_name):

    xml_name = parse_xml_name(env_name)

    if "pusher" in env_name.lower() or "striker" in env_name.lower() \
        or "thrower" in env_name.lower():
        update_env(env_name)

    else:
        with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

            new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
            for line in f.readlines():
                if "gear" in line:
                    pattern = re.compile(r'(?<=gear=")\d+\.?\d*')
                    a = pattern.findall(line)
                    current_num = float(a[0])
                    replace_num = current_num * variety_degree
                    sub_result = re.sub(pattern, str(replace_num), line)

                    new_f.write(sub_result)
                else:
                    new_f.write(line)

            new_f.close()

    os.system('cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_gravity(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    if "pusher" in env_name.lower() or "striker" in env_name.lower() \
        or "thrower" in env_name.lower():
        update_env(env_name)

    else:

        with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

            new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
            for line in f.readlines():
                if "gravity" in line:
                    pattern = re.compile(r"gravity=\"(.*?)\"")
                    a = pattern.findall(line)
                    friction_list = a[0].split(" ")
                    new_friction_list = []
                    for num in friction_list:
                        new_friction_list.append(variety_degree*float(num))

                    replace_num = " ".join(str(i) for i in new_friction_list)
                    replace_num = "gravity=\""+replace_num+"\""
                    sub_result = re.sub(pattern, str(replace_num), line)

                    new_f.write(sub_result)
                else:
                    new_f.write(line)

            new_f.close()

    os.system('cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    if "pusher" in env_name.lower() or "striker" in env_name.lower() \
        or "thrower" in env_name.lower():
        update_env(env_name)
    else:
        with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

            new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
            for line in f.readlines():
                if "density" in line:
                    pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                    a = pattern.findall(line)
                    current_num = float(a[0])
                    replace_num = current_num * variety_degree
                    sub_result = re.sub(pattern, str(replace_num), line)

                    new_f.write(sub_result)
                else:
                    new_f.write(line)

            new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    if "pusher" in env_name.lower() or "striker" in env_name.lower() \
        or "thrower" in env_name.lower():
        update_env(env_name)

    else:
        with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

            new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
            for line in f.readlines():
                if "friction" in line:
                    pattern = re.compile(r"friction=\"(.*?)\"")
                    a = pattern.findall(line)
                    friction_list = a[0].split(" ")
                    new_friction_list = []
                    for num in friction_list:
                        new_friction_list.append(variety_degree*float(num))

                    replace_num = " ".join(str(i) for i in new_friction_list)
                    replace_num = "friction=\""+replace_num+"\""
                    sub_result = re.sub(pattern, str(replace_num), line)

                    new_f.write(sub_result)
                else:
                    new_f.write(line)

            new_f.close()

    os.system('cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))
    time.sleep(0.2)


# -*- coding: utf-8 -*-
"""
@author: zhangc
"""

from xml.etree.ElementTree import ElementTree
import random


def read_xml(in_path):
    '''Read and parse the XML file.
      in_path: xml path
      return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def write_xml(tree, out_path):
    '''Write the XML file.
      tree: xml tree
      out_path: Write-out path'''
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def if_match(node, kv_map):
    '''Determine whether a node contains all incoming parameter properties.
      node: node
      kv_map: Map of attributes and attribute values'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


# ---------------search -----
def find_nodes(tree, path):
    '''Find all the nodes that match a path.
      tree: xml tree
      path: Node path'''
    return tree.findall(path)


def get_node_by_keyvalue(nodelist, kv_map):
    '''Locate the corresponding node according to the attribute and attribute value,
       and return the node.
      nodelist: Node list
      kv_map: Matching property and attribute value map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


# ---------------change -----
def change_node_properties(nodelist, kv_map, is_delete=False):
    '''Modify / add / delete node's property and attribute values.
      nodelist: Node list
      kv_map:Attribute and attribute value map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))


def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''Changing / adding / deleting the text of a node
      nodelist:Node list
      text : Updated text'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text


def update_env(env_name):
    if "pusher" in env_name.lower():

        tree = read_xml("./xml_path/source_file/pusher.xml")
        nodes_geom_0 = find_nodes(tree, "worldbody/body/body/body/body/geom")
        nodes_body_0 = find_nodes(tree, "worldbody/body/body/body/body/body")
        nodes_geom_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/geom")
        nodes_body_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/body")
        random_0 = random.uniform(0.2, 0.4)
        random_1 = random.uniform(0.3, 0.5)
        result_nodes_geom0 = get_node_by_keyvalue(nodes_geom_0, {"name": "ua"})
        change_node_properties(result_nodes_geom0, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_1, 0, 0)})

        result_nodes_body0 = get_node_by_keyvalue(nodes_body_0, {"name": "r_elbow_flex_link"})
        change_node_properties(result_nodes_body0, {"pos": "%f %f %f" % (random_1, 0, 0)})

        result_nodes_geom1 = get_node_by_keyvalue(nodes_geom_1, {"name": "fa"})
        change_node_properties(result_nodes_geom1, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_0, 0, 0)})

        result_nodes_body1 = get_node_by_keyvalue(nodes_body_1, {"name": "r_wrist_flex_link"})
        change_node_properties(result_nodes_body1, {"pos": "%f %f %f" % (random_0 + 0.03, 0, 0)})
        write_xml(tree, "./xml_path/target_file/pusher.xml")

    elif "striker" in env_name.lower():

        tree = read_xml("./xml_path/source_file/striker.xml")
        nodes_geom_0 = find_nodes(tree, "worldbody/body/body/body/body/geom")
        nodes_body_0 = find_nodes(tree, "worldbody/body/body/body/body/body")
        nodes_geom_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/geom")
        nodes_body_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/body")
        random_0 = random.uniform(0.2, 0.4)
        random_1 = random.uniform(0.3, 0.5)
        result_nodes_geom0 = get_node_by_keyvalue(nodes_geom_0, {"name": "ua"})
        change_node_properties(result_nodes_geom0, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_1, 0, 0)})

        result_nodes_body0 = get_node_by_keyvalue(nodes_body_0, {"name": "r_elbow_flex_link"})
        change_node_properties(result_nodes_body0, {"pos": "%f %f %f" % (random_1, 0, 0)})

        result_nodes_geom1 = get_node_by_keyvalue(nodes_geom_1, {"name": "fa"})
        change_node_properties(result_nodes_geom1, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_0, 0, 0)})

        result_nodes_body1 = get_node_by_keyvalue(nodes_body_1, {"name": "r_wrist_flex_link"})
        change_node_properties(result_nodes_body1, {"pos": "%f %f %f" % (random_0 + 0.03, 0, 0)})
        write_xml(tree, "./xml_path/target_file/striker%d.xml")

    elif "thrower" in env_name.lower():

        tree = read_xml("./xml_path/source_file/thrower.xml")
        nodes_geom_0 = find_nodes(tree, "worldbody/body/body/body/body/geom")
        nodes_body_0 = find_nodes(tree, "worldbody/body/body/body/body/body")
        nodes_geom_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/geom")
        nodes_body_1 = find_nodes(tree, "worldbody/body/body/body/body/body/body/body/body")
        random_0 = random.uniform(0.2, 0.4)
        random_1 = random.uniform(0.3, 0.5)
        result_nodes_geom0 = get_node_by_keyvalue(nodes_geom_0, {"name": "ua"})
        change_node_properties(result_nodes_geom0, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_1, 0, 0)})

        result_nodes_body0 = get_node_by_keyvalue(nodes_body_0, {"name": "r_elbow_flex_link"})
        change_node_properties(result_nodes_body0, {"pos": "%f %f %f" % (random_1, 0, 0)})

        result_nodes_geom1 = get_node_by_keyvalue(nodes_geom_1, {"name": "fa"})
        change_node_properties(result_nodes_geom1, {"fromto": "%f %f %f %f %f %f" % (0, 0, 0, random_0, 0, 0)})

        result_nodes_body1 = get_node_by_keyvalue(nodes_body_1, {"name": "r_wrist_flex_link"})
        change_node_properties(result_nodes_body1, {"pos": "%f %f %f" % (random_0 + 0.03, 0, 0)})
        write_xml(tree, "./xml_path/target_file/thrower.xml")

    else:
        print("Invalid argument")