#!/usr/bin/env ruby

require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  ruby '~> 2.6.3'
  gem 'pry-byebug'
end

module Day1
  @@input = File.readlines('input1').map(&:to_i).freeze

  def self.fuel_req(masses = @@input)
    masses.map { |i| i / 3 - 2 }.reduce(:+)
  end

  def self.fuel_for_fuel(masses = @@input)
    total_fuel = 0

    until masses.empty?
      masses = masses.map { |i| i / 3 - 2 }.select(&:positive?)
      total_fuel += masses.reduce(0, :+)
    end

    total_fuel
  end
end

module Day2
  @@input = File.read('input2').split(',').map(&:to_i).freeze

  def self.exec(intcode = @@input, input = 1202)
    intcode = intcode.dup
    intcode[1] = input / 100 # noun
    intcode[2] = input % 100 # verb

    ptr = 0
    ops = { 1 => :+, 2 => :* }

    loop do
      opcode, x_ref, y_ref, out = intcode[ptr, 4]
      break if opcode == 99

      op = ops.fetch(opcode)
      intcode[out] = intcode[x_ref].send(op, intcode[y_ref])
      ptr += 4
    end

    intcode.first
  end

  def self.find_input(intcode = @@input, output = 19690720)
    (0..9999).each do |input|
      out = compute(intcode, input) rescue -1
      return input if out == output
    end
  end
end

module Day3
  require 'matrix'

  @@input = File.readlines('input3').map do |path|
    path.split(',').map do |dir_dist|
      [dir_dist[0].to_sym, dir_dist[1..-1].to_i].freeze
    end.freeze
  end.freeze

  @@vectors = { U: Vector[1, 0], D: Vector[-1, 0], L: Vector[0, -1], R: Vector[0, 1] }

  def self.wirecross_dist(wirepaths = @@input)
    first_pos_steps = walk_path(wirepaths.first).to_h

    walk_path(wirepaths.last).reduce(Float::INFINITY) do |closest_cross, (pos, steps)|
      cross_dist = first_pos_steps.include?(pos) ? pos[0].abs + pos[1].abs : Float::INFINITY
      [closest_cross, cross_dist].min
    end
  end

  def self.wirecross_steps(wirepaths = @@input)
    first_pos_steps = walk_path(wirepaths.first).to_h

    walk_path(wirepaths.last).reduce(Float::INFINITY) do |closest_cross, (pos, steps)|
      cross_dist = first_pos_steps.fetch(pos, Float::INFINITY) + steps
      [closest_cross, cross_dist].min
    end
  end

  def self.walk_path(wirepath)
    return enum_for(:walk_path, wirepath) unless block_given?

    pos = Vector[0, 0]
    position_steps = Hash.new
    unit_steps = wirepath.flat_map { |dir, dist| Array.new(dist, dir) }

    unit_steps.each.with_index(1) do |step_dir, step_num|
      pos += @@vectors[step_dir]
      position_steps[pos] ||= step_num
      yield pos, position_steps[pos]
    end
  end
end

module Day4
  @@input = Range.new(*'108457-562041'.split('-').map(&:to_i))

  def self.valid_nums(range = @@input)
    range.count do |n|
      valid = (0..4).any? { |e| (((n / 10**e) % 100) % 11).zero? } # two adjacent digits are the same
      valid &&= n.digits.reverse == n.digits.sort
    end
  end

  def self.valid_nums_strict_pairs(range = @@input)
    range.count do |n|
      valid = n.digits.reverse == n.digits.sort
      valid &&= n.digits.chunk(&:itself).any? { |_v, vals| vals.size == 2 } # has a pair that isn't part of a larger group of matching digits
    end
  end
end

module Day5
  @@input = File.read('input5').split(',').map(&:to_i).freeze

  @@ops   = { add: 1, mult: 2, input: 3, out: 4, jit: 5, jif: 6, lt: 7, eq: 8, halt: 99 }.invert
  @@arity = { add: 3, mult: 3, input: 1, out: 1, jit: 2, jif: 2, lt: 3, eq: 3, halt: 0  }.transform_keys(&@@ops.invert)

  def self.test_air_conditioner
    run(@@input.dup, 1).find { |out| !out.zero? }
  end

  def self.test_thermal_radiator
    run(@@input.dup, 5).first
  end

  def self.run(intcode, input)
    return enum_for(:run, intcode, input) unless block_given?
    ptr = 0

    loop do
      modes, opcode = intcode[ptr].divmod(100)
      raw_params = intcode[ptr + 1, @@arity[opcode]]

      a, b, c = raw_params.each_with_index.map do |raw, i|
        is_position = modes.digits.fetch(i, 0).zero?
        OpenStruct.new(raw: raw, val: is_position ? intcode[raw] : raw)
      end

      jump_to = nil
      case @@ops[opcode]
      when :add   then intcode[c.raw] = a.val + b.val
      when :mult  then intcode[c.raw] = a.val * b.val
      when :input then intcode[a.raw] = input
      when :out   then yield a.val
      when :jit   then jump_to = b.val unless a.val.zero?
      when :jif   then jump_to = b.val if     a.val.zero?
      when :lt    then intcode[c.raw] = a.val < b.val ? 1 : 0
      when :eq    then intcode[c.raw] = a.val == b.val ? 1 : 0
      when :halt  then break
      end

      ptr = jump_to || ptr + 1 + @@arity[opcode]
    end
  end
end

module Day6
  @@input = File.read('input6').split.map do |orbit|
    orbit.split(')').reverse.each(&:freeze)
  end.to_h.freeze

  def self.num_orbits(orbits = @@input)
    bfs_distances(orbits, 'COM').each_value.reduce(:+)
  end

  def self.num_transfers(orbits = @@input)
    bfs_distances(orbits, 'YOU')['SAN'] - 2
  end

  def self.bfs_distances(edges, start)
    neighbors = Hash.new
    edges.each do |from, to|
      (neighbors[from] ||= []) << to
      (neighbors[to] ||= []) << from
    end

    distances = { start => 0 }
    deque = [start]
    until deque.empty?
      node = deque.shift
      neighbors[node].each do |n|
        deque << n unless distances.include?(n)
        distances[n] ||= distances[node] + 1
      end
    end
    distances
  end
end
