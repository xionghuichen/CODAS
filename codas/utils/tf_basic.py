import tensorflow as tf
from RLA.easy_log import logger
from abc import ABC, abstractmethod


class TfBasicClass(ABC):

    def __init__(self, scope):
        self.scope = scope
        self._global_scope = None
        self.should_reuse = False
        self.copy_weight_list = []

    def obj_graph_construct(self, source_input, *args, **kwargs):
        print_vars = True
        if self._global_scope is None:
            gsc = tf.variable_scope(self.scope)
        else:
            gsc = self.get_global_scope()

        with gsc:
            # if self._global_scope is not None:
            #     assert tf.get_variable_scope().name == self._global_scope, "conflict variable scope in the same tf object. {}, {}".format(tf.get_variable_scope().name, self._global_scope)
            if self.should_reuse:
                logger.info("reuse tf ops: {}".format(self.global_scope_name))
                # print_vars = False
            else:
                self.should_reuse = True
            if self._global_scope is None:
                self._global_scope = tf.get_variable_scope()
            ret_graph = self._obj_construct(source_input, *args, **kwargs)
            # if print_vars:
            #     for var in self.trainable_variables():
            #         logger.log(var)
            return ret_graph

    @abstractmethod
    def _obj_construct(self, source_input, *args, **kwargs):
        raise NotImplementedError

    def get_global_scope(self):
        assert self._global_scope is not None
        return tf.variable_scope(self._global_scope, reuse=True)

    def _inner_ph_builder(self):
        raise NotImplementedError

    @property
    def global_scope_name(self):
        return self._global_scope.name

    def trainable_variables(self, filter=''):
        return self._variables(tf.GraphKeys.TRAINABLE_VARIABLES, filter=filter)

    def global_variables(self, filter=''):
        return self._variables(tf.GraphKeys.GLOBAL_VARIABLES, filter=filter)

    def _variables(self, key, filter=''):
        if self.global_scope_name is None:
            logger.info("[WARNING] tf obj has not been created yet. in scope {}".format(self.scope))
            return []
        else:
            return tf.get_collection(key, self.global_scope_name + '/' + filter)

    def copy_weights(self, source_obj, sess=None):
        assert isinstance(source_obj, TfBasicClass)
        if source_obj.global_scope_name in self.copy_weight_list:
            logger.info("reuse tf ops")
            tf.get_variable_scope().reuse_variables()
        else:
            self.copy_weight_list.append(source_obj.global_scope_name)
        assert len(self.trainable_variables) == len(source_obj.trainable_variables)
        updates = []
        for var, source_var in zip(self.trainable_variables, source_obj.trainable_variables):
            updates.append(tf.assign(var, source_var))
        ops = tf.group(*updates)
        if sess is not None:
            sess.run(ops)
        return ops