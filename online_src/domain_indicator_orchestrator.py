import numpy as np


class DomainIndicator:
    def __init__(
        self,
        domain_indicator,
        limit,
        threshold_src,
        threshold,
        base_iters,
        dynamic_dacs,
        max_iters,
        initial_lr,
        policy_lr,
        max_lr,
        threshold_max,
        reduce_training,
        far_domain,
        lr_far_domain,
    ):
        self.domain_indicator = domain_indicator
        self.limit = limit
        self.mem = []
        self.threshold_src = threshold_src
        self.threshold_max = threshold_max
        self.threshold_up, self.threshold_down = threshold
        self.domain = 0
        self.domains = []
        self.prev = None
        self.losses = []
        self.dynamic_dacs = dynamic_dacs
        self.dacs = 0.5 if not self.dynamic_dacs else self.dynamic_dacs[0]
        self.iters = self.iters_train = 0
        self.base_iters = base_iters
        self.log_jump = 0
        self.MAX_ITERS_TRAIN = max_iters
        self.initial_lr = initial_lr
        self.policy_lr = policy_lr
        self.lr = initial_lr
        self.max_lr = max_lr
        self.lr_far_domain = lr_far_domain
        self.min_reduce, self.max_reduce = reduce_training
        self.far_domain = far_domain
        self.last_domain = None

    def _is_source(self):
        return self.avg() <= self.threshold_src

    def is_far_domain(self):
        val = self.get_domain_value()
        return val >= self.far_domain

    def get_domain_value(self):
        return self.avg() if self.mem else self.last_domain

    def linear_interpolation(self, x, vals, dire):
        return (
            np.interp(x, [self.threshold_src, self.threshold_max], vals)
            if dire < 0
            else 1
        )

    def get_domain(self):
        return self.domain

    def get_lr(self):
        return self.lr

    def add(self, val):
        self.mem.append(val)

    def avg(self):
        return np.mean(self.mem)

    def _calculate_lr(self, val):
        diff = abs(val - self.prev)
        ratio = diff / self.threshold_up
        return min(2 * ratio * self.initial_lr, self.max_lr)

    def _jump(self, old, new):
        diff = new - old
        return diff > self.threshold_up or diff < self.threshold_down

    def _calculate_iters_train(self, val, iters):
        diff = abs(val - self.prev)
        ratio = diff / self.threshold_up
        return int(ratio * iters)

    def _update_args(self, changed, dire, val):
        if not changed:
            return
        # if we are going to a higher intensity use less source in DACS
        if self.dynamic_dacs:
            min_ratio, max_ratio = self.dynamic_dacs
            self.dacs = self.linear_interpolation(val, [min_ratio, max_ratio], -1)

        iters = self.base_iters * self.linear_interpolation(
            val, [self.min_reduce, self.max_reduce], dire
        )
        if self.policy_lr != "adaptive_init":
            self.iters_train = self._calculate_iters_train(val, iters) + max(
                self.iters_train - self.iters, 0
            )
            self.iters_train = min(self.iters_train, self.MAX_ITERS_TRAIN)
        else:
            self.lr = self._calculate_lr(val)
            self.iters_train = iters
        self.iters = 0

    def domain_changed(self):
        self.log_jump = 0
        if len(self.mem) < self.limit or not self.domain_indicator:
            return False, None

        val = self.avg()
        self.losses.append(val)

        if self.prev is None:
            self.prev = val
            return False, None

        changed, dire = False, None
        if self._jump(self.prev, val):
            changed, dire = True, np.sign(val - self.prev)
            self._update_args(changed, dire, val)
            self.log_jump = self.prev - val
            self.prev = None
            self.domain += dire

        self.mem = []
        self.last_domain = val
        return changed, dire

    def is_training(self):
        if not self.domain_indicator:
            return True
        return self.iters < self.iters_train

    def get_args(self):
        if not self.domain_indicator:
            return {}

        train = self.is_training()
        self.iters += 1
        return dict(
            dacs=self.dacs,
            train=train,
        )
